#include "DIF/DenseMapping.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

#include <opencv2/imgproc.hpp>

#include "KeyFrame.h"

namespace ORB_SLAM3 {

DenseMapping::DenseMapping(const DenseMappingConfig& cfg)
    : mCfg(cfg)
{
}

DenseMapping::~DenseMapping()
{
    Stop();
}

void DenseMapping::SetConfig(const DenseMappingConfig& cfg)
{
    mCfg = cfg;
}

void DenseMapping::Start()
{
    if(mbRunning)
        return;
    mbStop = false;
    mbRunning = true;
    mThread = std::thread(&DenseMapping::ThreadMain, this);
}

void DenseMapping::Stop()
{
    if(!mbRunning)
        return;
    mbStop = true;
    mCvQueue.notify_all();
    if(mThread.joinable())
        mThread.join();
    mbRunning = false;
}

void DenseMapping::Reset()
{
    {
        std::lock_guard<std::mutex> lock(mMutexQueue);
        mQueue.clear();
    }
    {
        std::lock_guard<std::mutex> lock(mMutexData);
        mKeyFrameClouds.clear();
        mIntegratedKFIds.clear();
        mSegFrameClouds.clear();
        mGlobalVoxels.clear();
        mTrackToVoxels.clear();
        mTrackGeneration.clear();
    }
}

void DenseMapping::Enqueue(DenseKFPacket&& packet)
{
    if(!mCfg.enable)
        return;
    std::lock_guard<std::mutex> lock(mMutexQueue);
    if(mCfg.queue_max > 0)
    {
        while(static_cast<int>(mQueue.size()) >= mCfg.queue_max)
        {
            // Never drop rollback events; drop the oldest normal frame packet instead.
            auto it = std::find_if(mQueue.begin(), mQueue.end(), [](const DenseKFPacket& p) {
                return p.kind == DensePacketKind::Frame;
            });
            if(it == mQueue.end())
                break;
            mQueue.erase(it);
        }
    }
    mQueue.emplace_back(std::move(packet));
    mCvQueue.notify_one();
}

void DenseMapping::EnqueueTrackEnterD(int track_id, int frame_id, double timestamp)
{
    if(!mCfg.enable)
        return;
    if(track_id < 0)
        return;
    DenseKFPacket ev;
    ev.kind = DensePacketKind::TrackEnterD;
    ev.event_track_id = track_id;
    ev.frame_id = frame_id;
    ev.timestamp = timestamp;
    Enqueue(std::move(ev));
}

bool DenseMapping::SaveDensePointCloud(const std::string& path, bool use_latest_pose)
{
    std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> voxels;
    std::unordered_map<int, std::unordered_set<VoxelKey, VoxelKeyHash>> track_to_voxels;
    BuildVoxelMapFromClouds(use_latest_pose, voxels, track_to_voxels);
    return ExportVoxelsToPLY(path, voxels);
}

void DenseMapping::RebuildGlobalVoxels(bool use_latest_pose)
{
    std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> voxels;
    std::unordered_map<int, std::unordered_set<VoxelKey, VoxelKeyHash>> track_to_voxels;
    BuildVoxelMapFromClouds(use_latest_pose, voxels, track_to_voxels);
    std::lock_guard<std::mutex> lock(mMutexData);
    mGlobalVoxels.swap(voxels);
    mTrackToVoxels.swap(track_to_voxels);
}

void DenseMapping::ThreadMain()
{
    while(true)
    {
        DenseKFPacket packet;
        {
            std::unique_lock<std::mutex> lock(mMutexQueue);
            mCvQueue.wait(lock, [&]() { return mbStop || !mQueue.empty(); });
            if(mbStop && mQueue.empty())
                break;
            packet = std::move(mQueue.front());
            mQueue.pop_front();
        }
        ProcessPacket(packet);
    }
}

void DenseMapping::ProcessPacket(DenseKFPacket& packet)
{
    if(packet.kind == DensePacketKind::TrackEnterD)
    {
        if(mCfg.defusion_enable)
            DefuseTrack(packet.event_track_id);
        return;
    }

    if(!packet.has_pose)
        return;
    if(packet.label_map.empty() || packet.label_map.type() != CV_16S)
        return;
    if(packet.depth.empty())
        return;
    if(packet.local_to_global.empty())
        return;

    cv::Mat depth = packet.depth;
    if(depth.type() != CV_32F)
    {
        depth.convertTo(depth, CV_32F);
    }
    else if(mCfg.depth_median_ksize >= 3)
    {
        // Avoid mutating shared depth buffers held by Tracking caches.
        depth = depth.clone();
    }

    if(mCfg.depth_median_ksize >= 3)
    {
        int k = mCfg.depth_median_ksize;
        if(k % 2 == 0)
            k += 1;
        k = std::max(3, std::min(k, 31));
        cv::medianBlur(depth, depth, k);
    }

    cv::Mat bgr;
    if(!ConvertRgbToBgrIfNeeded(packet, bgr))
        return;

    if(depth.size() != packet.label_map.size() || bgr.size() != packet.label_map.size())
        return;

    if(packet.fx <= 0.0f || packet.fy <= 0.0f)
        return;

    cv::Mat forbid = BuildForbidMask(packet);
    cv::Mat unsafe = BuildUnsafeMaskFromLabelMap(packet.label_map);

    const Sophus::SE3f Twc = packet.Tcw.inverse();
    const float inv_fx = 1.0f / packet.fx;
    const float inv_fy = 1.0f / packet.fy;

    const int rows = packet.label_map.rows;
    const int cols = packet.label_map.cols;
    const int stride = std::max(1, mCfg.stride);

    const float edge_delta_m = mCfg.depth_edge_delta_m;
    const float edge_delta_ratio = mCfg.depth_edge_delta_ratio;
    const int edge_min_neighbors = std::max(0, mCfg.depth_edge_min_neighbors);
    const bool edge_reject_enable = (edge_delta_m > 0.0f || edge_delta_ratio > 0.0f);

    DenseKeyFrameCloud cloud;
    cloud.kf_id = packet.kf_id;
    cloud.frame_id = packet.frame_id;
    cloud.timestamp = packet.timestamp;
    cloud.pKF = packet.pKF;
    cloud.Tcw = packet.Tcw;

    {
        std::lock_guard<std::mutex> lock(mMutexData);
        if(packet.kf_id >= 0 && mIntegratedKFIds.find(packet.kf_id) != mIntegratedKFIds.end())
            return;
    }

    {
        std::lock_guard<std::mutex> lock(mMutexData);
        for(int y = 0; y < rows; y += stride)
        {
            const int16_t* label_ptr = packet.label_map.ptr<int16_t>(y);
            const float* depth_ptr = depth.ptr<float>(y);
            const cv::Vec3b* bgr_ptr = bgr.ptr<cv::Vec3b>(y);
            const uchar* forbid_ptr = forbid.ptr<uchar>(y);
            const uchar* unsafe_ptr = unsafe.empty() ? nullptr : unsafe.ptr<uchar>(y);
            for(int x = 0; x < cols; x += stride)
            {
                if(forbid_ptr[x])
                    continue;
                if(unsafe_ptr && unsafe_ptr[x])
                    continue;

                const int label = static_cast<int>(label_ptr[x]);

                const float z = depth_ptr[x];
                if(!std::isfinite(z) || z <= 0.0f)
                    continue;
                if(z < mCfg.depth_min || z > mCfg.depth_max)
                    continue;
                if(edge_reject_enable)
                {
                    const float thr = std::max(edge_delta_m, edge_delta_ratio * z);
                    if(thr > 0.0f)
                    {
                        float zmin = z;
                        float zmax = z;
                        int n_valid = 1;
                        for(int yy = std::max(0, y - 1); yy <= std::min(rows - 1, y + 1); ++yy)
                        {
                            const float* dptr = depth.ptr<float>(yy);
                            for(int xx = std::max(0, x - 1); xx <= std::min(cols - 1, x + 1); ++xx)
                            {
                                if(xx == x && yy == y)
                                    continue;
                                const float zn = dptr[xx];
                                if(!std::isfinite(zn) || zn <= 0.0f)
                                    continue;
                                if(zn < mCfg.depth_min || zn > mCfg.depth_max)
                                    continue;
                                zmin = std::min(zmin, zn);
                                zmax = std::max(zmax, zn);
                                ++n_valid;
                            }
                        }
                        if(edge_min_neighbors > 0 && n_valid >= edge_min_neighbors)
                        {
                            if((zmax - zmin) > thr)
                                continue;
                        }
                    }
                }

                int instance_id = -1;
                float w_lock = 1.0f;
                uint32_t gen = 0;
                if(label >= 0)
                {
                    if(label >= static_cast<int>(packet.local_to_global.size()))
                        continue;
                    const int gid = packet.local_to_global[static_cast<size_t>(label)];
                    if(gid < 0)
                        continue;
                    auto it = packet.track_states.find(gid);
                    if(it == packet.track_states.end())
                        continue;
                    if(it->second.state == DIFTrackState::D)
                        continue;
                    instance_id = gid;
                    w_lock = std::max(0.0f, it->second.w_lock);
                    auto git = mTrackGeneration.find(gid);
                    if(git != mTrackGeneration.end())
                        gen = git->second;
                }

                const float X = (static_cast<float>(x) - packet.cx) * z * inv_fx;
                const float Y = (static_cast<float>(y) - packet.cy) * z * inv_fy;
                const float Z = z;
                const Eigen::Vector3f pc(X, Y, Z);
                const Eigen::Vector3f pw = Twc * pc;

                const cv::Vec3b c = bgr_ptr[x];
                const Eigen::Vector3f rgb(static_cast<float>(c[2]),
                                          static_cast<float>(c[1]),
                                          static_cast<float>(c[0]));

                DensePoint p;
                p.pc = pc;
                p.rgb = rgb;
                p.instance_id = instance_id;
                p.w_lock = w_lock;
                p.gen = gen;
                cloud.points.emplace_back(p);

                IntegratePoint(pw, rgb, instance_id, w_lock, gen, packet.frame_id);
            }
        }

        if(packet.kf_id >= 0)
        {
            mIntegratedKFIds.insert(packet.kf_id);
            mKeyFrameClouds[packet.kf_id] = std::move(cloud);
        }
        else
        {
            mSegFrameClouds.emplace_back(std::move(cloud));
        }
    }
}

cv::Mat DenseMapping::BuildForbidMask(const DenseKFPacket& packet) const
{
    cv::Mat forbid(packet.label_map.size(), CV_8U, cv::Scalar(0));
    const int rows = packet.label_map.rows;
    const int cols = packet.label_map.cols;

    const bool fill_bbox = mCfg.forbid_fill_bbox && !packet.local_to_global.empty();
    std::vector<int> bbox_min_x;
    std::vector<int> bbox_min_y;
    std::vector<int> bbox_max_x;
    std::vector<int> bbox_max_y;
    if(fill_bbox)
    {
        const size_t n = packet.local_to_global.size();
        bbox_min_x.assign(n, cols);
        bbox_min_y.assign(n, rows);
        bbox_max_x.assign(n, -1);
        bbox_max_y.assign(n, -1);
    }

    for(int y = 0; y < rows; ++y)
    {
        const int16_t* label_ptr = packet.label_map.ptr<int16_t>(y);
        uchar* forbid_ptr = forbid.ptr<uchar>(y);
        for(int x = 0; x < cols; ++x)
        {
            const int label = static_cast<int>(label_ptr[x]);
            if(label < 0)
                continue;

            if(label >= static_cast<int>(packet.local_to_global.size()))
            {
                forbid_ptr[x] = 255;
                continue;
            }

            const int gid = packet.local_to_global[static_cast<size_t>(label)];
            if(gid < 0)
            {
                forbid_ptr[x] = 255;
                continue;
            }

            const auto it = packet.track_states.find(gid);
            if(it == packet.track_states.end())
            {
                forbid_ptr[x] = 255;
                continue;
            }

            // DIF-SLAM v2.0: forbid only when state == D (map_lock must not forbid mapping).
            const bool forbid_inst = (it->second.state == DIFTrackState::D);
            if(forbid_inst)
            {
                forbid_ptr[x] = 255;
                if(fill_bbox)
                {
                    bbox_min_x[static_cast<size_t>(label)] = std::min(bbox_min_x[static_cast<size_t>(label)], x);
                    bbox_max_x[static_cast<size_t>(label)] = std::max(bbox_max_x[static_cast<size_t>(label)], x);
                    bbox_min_y[static_cast<size_t>(label)] = std::min(bbox_min_y[static_cast<size_t>(label)], y);
                    bbox_max_y[static_cast<size_t>(label)] = std::max(bbox_max_y[static_cast<size_t>(label)], y);
                }
            }
        }
    }

    if(fill_bbox)
    {
        const int margin = std::max(0, mCfg.forbid_bbox_margin_px);
        const size_t n = bbox_min_x.size();
        for(size_t i = 0; i < n; ++i)
        {
            if(bbox_max_x[i] < 0 || bbox_max_y[i] < 0)
                continue;
            int x1 = bbox_min_x[i] - margin;
            int y1 = bbox_min_y[i] - margin;
            int x2 = bbox_max_x[i] + margin;
            int y2 = bbox_max_y[i] + margin;
            x1 = std::max(0, std::min(cols - 1, x1));
            y1 = std::max(0, std::min(rows - 1, y1));
            x2 = std::max(0, std::min(cols - 1, x2));
            y2 = std::max(0, std::min(rows - 1, y2));
            if(x2 < x1 || y2 < y1)
                continue;
            forbid(cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1)).setTo(255);
        }
    }

    if(!packet.forbid_mask.empty() && packet.forbid_mask.size() == forbid.size())
    {
        if(packet.forbid_mask.type() == CV_8U)
        {
            cv::bitwise_or(forbid, packet.forbid_mask, forbid);
        }
        else
        {
            cv::Mat tmp;
            packet.forbid_mask.convertTo(tmp, CV_8U);
            cv::bitwise_or(forbid, tmp, forbid);
        }
    }

    if(mCfg.forbid_dilate_px > 0)
    {
        const int k = std::max(1, mCfg.forbid_dilate_px);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * k + 1, 2 * k + 1));
        cv::dilate(forbid, forbid, kernel);
    }

    return forbid;
}

cv::Mat DenseMapping::BuildUnsafeMaskFromLabelMap(const cv::Mat& label_map) const
{
    if(mCfg.instance_guard_px <= 0)
        return cv::Mat();
    if(label_map.empty() || label_map.type() != CV_16S)
        return cv::Mat();

    const int rows = label_map.rows;
    const int cols = label_map.cols;
    if(rows <= 1 || cols <= 1)
        return cv::Mat();

    cv::Mat boundary(rows, cols, CV_8U, cv::Scalar(0));
    for(int y = 0; y < rows; ++y)
    {
        const int16_t* row = label_map.ptr<int16_t>(y);
        const int16_t* row_up = (y > 0) ? label_map.ptr<int16_t>(y - 1) : nullptr;
        const int16_t* row_dn = (y + 1 < rows) ? label_map.ptr<int16_t>(y + 1) : nullptr;
        uchar* bout = boundary.ptr<uchar>(y);
        for(int x = 0; x < cols; ++x)
        {
            const int16_t v = row[x];
            bool diff = false;
            if(x > 0 && row[x - 1] != v)
                diff = true;
            else if(x + 1 < cols && row[x + 1] != v)
                diff = true;
            else if(row_up && row_up[x] != v)
                diff = true;
            else if(row_dn && row_dn[x] != v)
                diff = true;
            if(diff)
                bout[x] = 255;
        }
    }

    const int k = std::max(1, mCfg.instance_guard_px);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * k + 1, 2 * k + 1));
    cv::Mat unsafe;
    cv::dilate(boundary, unsafe, kernel);
    return unsafe;
}

bool DenseMapping::ConvertRgbToBgrIfNeeded(const DenseKFPacket& packet, cv::Mat& out_bgr) const
{
    if(packet.rgb.empty())
        return false;

    if(packet.rgb.type() == CV_8UC3)
    {
        if(packet.is_rgb)
            cv::cvtColor(packet.rgb, out_bgr, cv::COLOR_RGB2BGR);
        else
            out_bgr = packet.rgb;
        return true;
    }

    if(packet.rgb.type() == CV_8UC1)
    {
        cv::cvtColor(packet.rgb, out_bgr, cv::COLOR_GRAY2BGR);
        return true;
    }

    if(packet.rgb.type() == CV_8UC4)
    {
        if(packet.is_rgb)
            cv::cvtColor(packet.rgb, out_bgr, cv::COLOR_RGBA2BGR);
        else
            cv::cvtColor(packet.rgb, out_bgr, cv::COLOR_BGRA2BGR);
        return true;
    }

    return false;
}

void DenseMapping::IntegratePoint(const Eigen::Vector3f& pw,
                                  const Eigen::Vector3f& rgb,
                                  int instance_id,
                                  float w_lock,
                                  uint32_t gen,
                                  int frame_id)
{
    if(mCfg.voxel_size_m <= 0.0f)
        return;
    if(!(w_lock > 0.0f) || !std::isfinite(w_lock))
        return;
    const float inv = 1.0f / mCfg.voxel_size_m;
    const int vx = static_cast<int>(std::floor(pw.x() * inv));
    const int vy = static_cast<int>(std::floor(pw.y() * inv));
    const int vz = static_cast<int>(std::floor(pw.z() * inv));
    VoxelKey key(vx, vy, vz);
    VoxelAccum& v = mGlobalVoxels[key];
    v.sum_p += w_lock * pw;
    v.sum_rgb += w_lock * rgb;
    v.w_total += w_lock;
    if(v.last_obs_frame_id != frame_id)
    {
        v.last_obs_frame_id = frame_id;
        v.obs_frames += 1;
    }

    if(mCfg.defusion_enable && instance_id >= 0)
    {
        // Track->voxel reverse index for rollback.
        mTrackToVoxels[instance_id].insert(key);

        // Per-voxel per-track contribution for rollback.
        bool found = false;
        for(TrackAccum& c : v.contrib)
        {
            if(c.gid == instance_id)
            {
                c.sum_p += w_lock * pw;
                c.sum_rgb += w_lock * rgb;
                c.w += w_lock;
                found = true;
                break;
            }
        }
        if(!found)
        {
            TrackAccum c;
            c.gid = instance_id;
            c.sum_p = w_lock * pw;
            c.sum_rgb = w_lock * rgb;
            c.w = w_lock;
            v.contrib.emplace_back(c);
        }
    }
    (void)gen;
}

void DenseMapping::DefuseTrack(int track_id)
{
    if(!mCfg.defusion_enable)
        return;
    if(track_id < 0)
        return;

    std::lock_guard<std::mutex> lock(mMutexData);

    const auto itset = mTrackToVoxels.find(track_id);
    if(itset != mTrackToVoxels.end())
    {
        for(const VoxelKey& key : itset->second)
        {
            auto itv = mGlobalVoxels.find(key);
            if(itv == mGlobalVoxels.end())
                continue;
            VoxelAccum& v = itv->second;

            for(size_t i = 0; i < v.contrib.size(); ++i)
            {
                if(v.contrib[i].gid != track_id)
                    continue;
                v.sum_p -= v.contrib[i].sum_p;
                v.sum_rgb -= v.contrib[i].sum_rgb;
                v.w_total -= v.contrib[i].w;
                v.contrib[i] = v.contrib.back();
                v.contrib.pop_back();
                break;
            }

            if(!(v.w_total > 1e-6f))
                mGlobalVoxels.erase(itv);
        }
        mTrackToVoxels.erase(itset);
    }

    // Bump generation so any cached points from older generations are ignored during rebuild/export.
    mTrackGeneration[track_id] = mTrackGeneration[track_id] + 1;
}

bool DenseMapping::ExportVoxelsToPLY(const std::string& path,
                                     const std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash>& voxels) const
{
    std::ofstream ofs(path);
    if(!ofs.is_open())
        return false;

    const int min_w = std::max(1, mCfg.min_voxel_weight);
    const int min_neighbors = std::max(0, mCfg.min_voxel_neighbors);
    const int min_frames = std::max(0, mCfg.min_voxel_frames);

    auto is_exportable = [&](const VoxelAccum& v) -> bool {
        if(v.w_total < static_cast<float>(min_w))
            return false;
        if(min_frames > 0 && v.obs_frames < min_frames)
            return false;
        return true;
    };

    auto has_neighbor_support = [&](const VoxelKey& k) -> bool {
        if(min_neighbors <= 0)
            return true;
        int cnt = 0;
        for(int dx = -1; dx <= 1; ++dx)
        {
            for(int dy = -1; dy <= 1; ++dy)
            {
                for(int dz = -1; dz <= 1; ++dz)
                {
                    if(dx == 0 && dy == 0 && dz == 0)
                        continue;
                    const VoxelKey nk(k.x + dx, k.y + dy, k.z + dz);
                    const auto it = voxels.find(nk);
                    if(it == voxels.end())
                        continue;
                    if(!is_exportable(it->second))
                        continue;
                    if(++cnt >= min_neighbors)
                        return true;
                }
            }
        }
        return false;
    };

    size_t n_export = 0;
    for(const auto& kv : voxels)
    {
        if(!is_exportable(kv.second))
            continue;
        if(!has_neighbor_support(kv.first))
            continue;
        ++n_export;
    }

    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << n_export << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "end_header\n";
    ofs << std::fixed << std::setprecision(6);

    for(const auto& kv : voxels)
    {
        const VoxelAccum& v = kv.second;
        if(!is_exportable(v))
            continue;
        if(!has_neighbor_support(kv.first))
            continue;
        const float w = std::max(1e-12f, v.w_total);
        const Eigen::Vector3f p = v.sum_p / w;
        const Eigen::Vector3f c = v.sum_rgb / w;
        const int r = std::max(0, std::min(255, static_cast<int>(std::lround(c.x()))));
        const int g = std::max(0, std::min(255, static_cast<int>(std::lround(c.y()))));
        const int b = std::max(0, std::min(255, static_cast<int>(std::lround(c.z()))));
        ofs << p.x() << " " << p.y() << " " << p.z() << " " << r << " " << g << " " << b << "\n";
    }

    return true;
}

void DenseMapping::BuildVoxelMapFromClouds(
    bool use_latest_pose,
    std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash>& out_voxels,
    std::unordered_map<int, std::unordered_set<VoxelKey, VoxelKeyHash>>& out_track_to_voxels) const
{
    out_voxels.clear();
    out_track_to_voxels.clear();
    if(mCfg.voxel_size_m <= 0.0f)
        return;

    std::lock_guard<std::mutex> lock(mMutexData);

    const float inv = 1.0f / mCfg.voxel_size_m;
    const auto get_gen = [&](int gid) -> uint32_t {
        const auto it = mTrackGeneration.find(gid);
        if(it == mTrackGeneration.end())
            return 0;
        return it->second;
    };

    auto integrate_point = [&](const Eigen::Vector3f& pw,
                               const Eigen::Vector3f& rgb,
                               int gid,
                               float w_lock,
                               int frame_id)
    {
        if(mCfg.voxel_size_m <= 0.0f)
            return;
        if(!(w_lock > 0.0f) || !std::isfinite(w_lock))
            return;

        const int vx = static_cast<int>(std::floor(pw.x() * inv));
        const int vy = static_cast<int>(std::floor(pw.y() * inv));
        const int vz = static_cast<int>(std::floor(pw.z() * inv));
        VoxelKey key(vx, vy, vz);
        VoxelAccum& v = out_voxels[key];
        v.sum_p += w_lock * pw;
        v.sum_rgb += w_lock * rgb;
        v.w_total += w_lock;
        if(v.last_obs_frame_id != frame_id)
        {
            v.last_obs_frame_id = frame_id;
            v.obs_frames += 1;
        }

        if(mCfg.defusion_enable && gid >= 0)
        {
            out_track_to_voxels[gid].insert(key);
            bool found = false;
            for(TrackAccum& c : v.contrib)
            {
                if(c.gid == gid)
                {
                    c.sum_p += w_lock * pw;
                    c.sum_rgb += w_lock * rgb;
                    c.w += w_lock;
                    found = true;
                    break;
                }
            }
            if(!found)
            {
                TrackAccum c;
                c.gid = gid;
                c.sum_p = w_lock * pw;
                c.sum_rgb = w_lock * rgb;
                c.w = w_lock;
                v.contrib.emplace_back(c);
            }
        }
    };

    auto integrate_cloud = [&](const DenseKeyFrameCloud& cloud, const Sophus::SE3f& Tcw)
    {
        const Sophus::SE3f Twc = Tcw.inverse();
        for(const DensePoint& p : cloud.points)
        {
            if(p.instance_id >= 0)
            {
                // Generation filter: after a rollback event, exclude points from older generations.
                if(p.gen != get_gen(p.instance_id))
                    continue;
            }
            const Eigen::Vector3f pw = Twc * p.pc;
            integrate_point(pw, p.rgb, p.instance_id, p.w_lock, cloud.frame_id);
        }
    };

    for(const auto& kv : mKeyFrameClouds)
    {
        const DenseKeyFrameCloud& cloud = kv.second;
        if(use_latest_pose && cloud.pKF)
            integrate_cloud(cloud, cloud.pKF->GetPose());
        else
            integrate_cloud(cloud, cloud.Tcw);
    }

    for(const auto& cloud : mSegFrameClouds)
    {
        integrate_cloud(cloud, cloud.Tcw);
    }
}

} // namespace ORB_SLAM3
