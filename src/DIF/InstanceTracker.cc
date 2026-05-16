#include "DIF/InstanceTracker.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "CameraModels/GeometricCamera.h"

namespace ORB_SLAM3 {
namespace {

constexpr double kInfCost = 1e9;

struct AssignmentResult
{
    // row -> col, size rows, -1 means unmatched
    std::vector<int> row_to_col;
};

// Rectangular Hungarian (min cost). costs: rows x cols, finite.
AssignmentResult HungarianMinimize(const std::vector<std::vector<double>>& costs)
{
    const int n = static_cast<int>(costs.size());
    const int m = (n > 0) ? static_cast<int>(costs[0].size()) : 0;
    AssignmentResult res;
    res.row_to_col.assign(n, -1);
    if(n == 0 || m == 0)
        return res;

    // Ensure n <= m by transposing if needed.
    bool transposed = false;
    std::vector<std::vector<double>> a;
    int nn = n, mm = m;
    if(n > m)
    {
        transposed = true;
        nn = m;
        mm = n;
        a.assign(nn, std::vector<double>(mm, 0.0));
        for(int i = 0; i < nn; ++i)
            for(int j = 0; j < mm; ++j)
                a[i][j] = costs[j][i];
    }
    else
    {
        a = costs;
    }

    // 1-indexed algorithm
    std::vector<double> u(nn + 1, 0.0), v(mm + 1, 0.0);
    std::vector<int> p(mm + 1, 0), way(mm + 1, 0);

    for(int i = 1; i <= nn; ++i)
    {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(mm + 1, std::numeric_limits<double>::infinity());
        std::vector<char> used(mm + 1, 0);
        do
        {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int j1 = 0;
            for(int j = 1; j <= mm; ++j)
            {
                if(used[j]) continue;
                const double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                if(cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if(minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for(int j = 0; j <= mm; ++j)
            {
                if(used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while(p[j0] != 0);

        do
        {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while(j0 != 0);
    }

    // p[j] matched row p[j] with col j
    if(!transposed)
    {
        for(int j = 1; j <= mm; ++j)
        {
            const int i = p[j];
            if(i >= 1 && i <= nn)
                res.row_to_col[i - 1] = j - 1;
        }
        return res;
    }

    // Transposed case: rows=original cols (m), cols=original rows (n).
    // For each original row r (column index j in transposed), p[j] tells which original col is assigned.
    for(int j = 1; j <= mm; ++j)
    {
        const int orig_row = j - 1;
        const int i = p[j]; // row in transposed => original col
        if(orig_row >= 0 && orig_row < n)
        {
            if(i >= 1 && i <= nn)
                res.row_to_col[orig_row] = i - 1;
            else
                res.row_to_col[orig_row] = -1;
        }
    }
    return res;
}

template <typename T>
T MedianInPlace(std::vector<T>& v)
{
    if(v.empty())
        return T();
    const size_t k = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

} // namespace

DIFInstanceTracker::DIFInstanceTracker(const DIFInstanceTrackerConfig& cfg) : mCfg(cfg) {}

void DIFInstanceTracker::SetConfig(const DIFInstanceTrackerConfig& cfg) { mCfg = cfg; }

void DIFInstanceTracker::Reset()
{
    mTracks.clear();
    mNextTrackId = 0;
}

int DIFInstanceTracker::AllocateNewTrackId() { return mNextTrackId++; }

float DIFInstanceTracker::GaussianScore(float d, float sigma)
{
    if(!(sigma > 1e-6f))
        return 0.0f;
    const float x = d / sigma;
    return std::exp(-0.5f * x * x);
}

bool DIFInstanceTracker::Compute3DCentroidMAD(
    const std::vector<Eigen::Vector3f>& points,
    int n_in_min,
    float mad_tau,
    Eigen::Vector3f& out_C,
    int* out_n_inliers,
    float* out_z_sigma_rob)
{
    if(out_n_inliers)
        *out_n_inliers = 0;
    if(out_z_sigma_rob)
        *out_z_sigma_rob = 0.0f;

    if(static_cast<int>(points.size()) < n_in_min)
        return false;

    // Robust depth clipping using Median + MAD on z (as in optimized DIF spec).
    std::vector<float> zs;
    zs.reserve(points.size());
    for(const auto& p : points)
        zs.push_back(p.z());
    const float z_med = MedianInPlace(zs);

    std::vector<float> abs_dev;
    abs_dev.reserve(zs.size());
    for(const float z : zs)
        abs_dev.push_back(std::abs(z - z_med));
    const float mad = MedianInPlace(abs_dev);
    const float sigma_rob = 1.4826f * std::max(mad, 1e-6f);
    const float thr = mad_tau * sigma_rob;
    if(out_z_sigma_rob)
        *out_z_sigma_rob = sigma_rob;

    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    int n_in = 0;
    for(const auto& p : points)
    {
        if(std::abs(p.z() - z_med) <= thr)
        {
            sum += p;
            n_in++;
        }
    }
    if(n_in < n_in_min)
    {
        if(out_n_inliers)
            *out_n_inliers = n_in;
        return false;
    }
    out_C = sum / static_cast<float>(n_in);
    if(out_n_inliers)
        *out_n_inliers = n_in;
    return true;
}

std::vector<DIFDetection> DIFInstanceTracker::BuildDetections(
    const cv::Mat& label_map,
    const cv::Mat& depth,
    const Sophus::SE3f& Tcw,
    GeometricCamera* camera,
    bool allow_3d_update,
    const std::vector<cv::Point2f>& keypoints_uv) const
{
    (void)Tcw;
    std::vector<DIFDetection> dets;
    if(label_map.empty() || label_map.type() != CV_16S)
        return dets;
    if(depth.empty() || depth.type() != CV_32F)
        return dets;
    if(label_map.rows != depth.rows || label_map.cols != depth.cols)
        return dets;

    const int H = label_map.rows;
    const int W = label_map.cols;

    int max_id = -1;
    {
        const int16_t* p = label_map.ptr<int16_t>(0);
        const int total = H * W;
        for(int i = 0; i < total; ++i)
            max_id = std::max(max_id, static_cast<int>(p[i]));
    }
    const int N = max_id + 1;
    if(N <= 0)
        return dets;

    std::vector<int> feat_count(static_cast<size_t>(N), 0);
    if(mCfg.n_feat_min > 0 && !keypoints_uv.empty())
    {
        for(const auto& pt : keypoints_uv)
        {
            const int x = static_cast<int>(pt.x + 0.5f);
            const int y = static_cast<int>(pt.y + 0.5f);
            if(x < 0 || y < 0 || x >= W || y >= H)
                continue;
            const int id = static_cast<int>(label_map.at<int16_t>(y, x));
            if(id >= 0 && id < N)
                feat_count[static_cast<size_t>(id)]++;
        }
    }

    std::vector<int> area(static_cast<size_t>(N), 0);
    std::vector<int> minx(static_cast<size_t>(N), W), miny(static_cast<size_t>(N), H);
    std::vector<int> maxx(static_cast<size_t>(N), -1), maxy(static_cast<size_t>(N), -1);
    std::vector<int64_t> sumx(static_cast<size_t>(N), 0), sumy(static_cast<size_t>(N), 0);

    for(int y = 0; y < H; ++y)
    {
        const int16_t* row = label_map.ptr<int16_t>(y);
        for(int x = 0; x < W; ++x)
        {
            const int id = static_cast<int>(row[x]);
            if(id < 0) continue;
            if(id >= N) continue;
            area[static_cast<size_t>(id)]++;
            minx[static_cast<size_t>(id)] = std::min(minx[static_cast<size_t>(id)], x);
            miny[static_cast<size_t>(id)] = std::min(miny[static_cast<size_t>(id)], y);
            maxx[static_cast<size_t>(id)] = std::max(maxx[static_cast<size_t>(id)], x);
            maxy[static_cast<size_t>(id)] = std::max(maxy[static_cast<size_t>(id)], y);
            sumx[static_cast<size_t>(id)] += x;
            sumy[static_cast<size_t>(id)] += y;
        }
    }

    if(mCfg.max_masks > 0)
        dets.reserve(static_cast<size_t>(std::min(N, mCfg.max_masks)));
    else
        dets.reserve(static_cast<size_t>(N));

    // Heuristic: allow larger instances to bypass the ORB feature coverage filter, so low-texture but relevant objects
    // (e.g., balloons) can still be tracked and produce v_obs, while very small fragments are suppressed.
    const int feat_relax_area = std::max(mCfg.area_min * 5, mCfg.area_min);

    for(int id = 0; id < N; ++id)
    {
        const int a = area[static_cast<size_t>(id)];
        if(a < mCfg.area_min)
            continue;
        if(mCfg.n_feat_min > 0 && feat_count[static_cast<size_t>(id)] < mCfg.n_feat_min && a < feat_relax_area)
            continue;

        DIFDetection det;
        det.local_id = id;
        det.area = a;
        det.bbox.x1 = minx[static_cast<size_t>(id)];
        det.bbox.y1 = miny[static_cast<size_t>(id)];
        det.bbox.x2 = maxx[static_cast<size_t>(id)];
        det.bbox.y2 = maxy[static_cast<size_t>(id)];
        if(!det.bbox.IsValid())
            continue;
        const int bw = det.bbox.Width();
        const int bh = det.bbox.Height();
        if(mCfg.bbox_min_w > 0 && bw < mCfg.bbox_min_w)
            continue;
        if(mCfg.bbox_min_h > 0 && bh < mCfg.bbox_min_h)
            continue;
        if(mCfg.fill_ratio_min > 0.0f)
        {
            const int bbox_area = bw * bh;
            if(bbox_area <= 0)
                continue;
            const float fill = static_cast<float>(det.area) / static_cast<float>(bbox_area);
            if(fill < mCfg.fill_ratio_min)
                continue;
        }
        det.c2d = cv::Point2f(static_cast<float>(sumx[static_cast<size_t>(id)]) / static_cast<float>(a),
                              static_cast<float>(sumy[static_cast<size_t>(id)]) / static_cast<float>(a));

        // Build bbox ROI mask
        det.mask.bbox = det.bbox;
        det.mask.mask = cv::Mat(det.bbox.Height(), det.bbox.Width(), CV_8U, cv::Scalar(0));
        for(int y = det.bbox.y1; y <= det.bbox.y2; ++y)
        {
            const int16_t* row = label_map.ptr<int16_t>(y);
            uint8_t* out = det.mask.mask.ptr<uint8_t>(y - det.bbox.y1);
            for(int x = det.bbox.x1; x <= det.bbox.x2; ++x)
                out[x - det.bbox.x1] = (row[x] == id) ? 1 : 0;
        }
        det.mask.area = det.area;

        if(allow_3d_update)
        {
            // 3D centroid (camera): sample inside bbox with stride, filter by label and depth.
            std::vector<Eigen::Vector3f> pts;
            pts.reserve(static_cast<size_t>(std::min(mCfg.sample_max_points, det.bbox.Area())));

            const float fx = camera ? camera->getParameter(0) : 0.f;
            const float fy = camera ? camera->getParameter(1) : 0.f;
            const float cx = camera ? camera->getParameter(2) : 0.f;
            const float cy = camera ? camera->getParameter(3) : 0.f;

            // For smaller instances, reduce sampling stride to improve the chance of collecting enough valid depth points.
            const bool small_instance = (det.area < feat_relax_area);
            const int stride0 = std::max(1, mCfg.sample_stride);
            const int stride = small_instance ? std::max(1, stride0 / 2) : stride0;

            auto sample_points = [&](int stride_s) {
                pts.clear();
                for(int y = det.bbox.y1; y <= det.bbox.y2; y += stride_s)
                {
                    const int16_t* row_l = label_map.ptr<int16_t>(y);
                    const float* row_d = depth.ptr<float>(y);
                    for(int x = det.bbox.x1; x <= det.bbox.x2; x += stride_s)
                    {
                        if(row_l[x] != id)
                            continue;
                        const float z = row_d[x];
                        if(!(z > mCfg.depth_min && z < mCfg.depth_max))
                            continue;
                        if(!std::isfinite(z))
                            continue;
                        if(!(fx > 1e-6f && fy > 1e-6f))
                            continue;
                        const float xc = (static_cast<float>(x) - cx) * z / fx;
                        const float yc = (static_cast<float>(y) - cy) * z / fy;
                        const Eigen::Vector3f Xc(xc, yc, z);
                        if(!std::isfinite(Xc.x()) || !std::isfinite(Xc.y()) || !std::isfinite(Xc.z()))
                            continue;
                        pts.push_back(Xc);
                        if(mCfg.sample_max_points > 0 && static_cast<int>(pts.size()) >= mCfg.sample_max_points)
                            return;
                    }
                }
            };

            sample_points(stride);

            Eigen::Vector3f Cc;
            int n_in = 0;
            float z_sigma = 0.0f;
            // Adaptive inlier threshold for small instances: require fewer depth samples to avoid missing thin/low-texture
            // dynamic objects, while keeping the original threshold for large instances.
            int n_in_min = mCfg.n_in_min;
            if(small_instance)
                n_in_min = std::min(n_in_min, std::max(10, det.area / 200));

            if(static_cast<int>(pts.size()) < n_in_min && stride > 1)
            {
                // Best-effort refinement: resample densely when too few valid depth points are collected.
                sample_points(1);
            }

            if(Compute3DCentroidMAD(pts, n_in_min, mCfg.mad_tau, Cc, &n_in, &z_sigma))
            {
                det.valid3d = true;
                det.Cc = Cc;
                det.n_inliers_3d = n_in;
                det.z_sigma_rob = z_sigma;
            }
        }
        dets.push_back(std::move(det));
    }

    // If the label_map has too many instances, keep the largest ones to protect Hungarian complexity,
    // rather than dropping the entire update (which would break track continuity).
    if(mCfg.max_masks > 0 && static_cast<int>(dets.size()) > mCfg.max_masks)
    {
        std::stable_sort(dets.begin(), dets.end(), [](const DIFDetection& a, const DIFDetection& b) {
            if(a.area != b.area)
                return a.area > b.area;
            return a.local_id < b.local_id;
        });
        dets.resize(static_cast<size_t>(mCfg.max_masks));
    }

    return dets;
}

bool DIFInstanceTracker::UpdateFromSegmentation(
    int frame_id,
    double timestamp,
    const cv::Mat& label_map,
    const cv::Mat& depth,
    const Sophus::SE3f& Tcw,
    GeometricCamera* camera,
    bool allow_3d_update,
    const std::vector<cv::Point2f>& keypoints_uv,
    std::vector<int>& out_local_to_global)
{
    out_local_to_global.clear();
    if(!mCfg.enable)
        return false;
    if(label_map.empty() || label_map.type() != CV_16S)
        return false;
    if(depth.empty() || depth.type() != CV_32F)
        return false;
    if(label_map.rows != depth.rows || label_map.cols != depth.cols)
        return false;

    const int image_area = label_map.rows * label_map.cols;
    const float fx_v = camera ? camera->getParameter(0) : 0.f;
    const float fy_v = camera ? camera->getParameter(1) : 0.f;
    const float f_avg = (fx_v > 1e-6f && fy_v > 1e-6f) ? (0.5f * (fx_v + fy_v)) : 0.0f;

    // Pre-size local->global for all local ids present in label_map (filtered detections keep -1).
    int max_local_all = -1;
    {
        const int16_t* p = label_map.ptr<int16_t>(0);
        const int total = label_map.rows * label_map.cols;
        for(int i = 0; i < total; ++i)
            max_local_all = std::max(max_local_all, static_cast<int>(p[i]));
    }
    out_local_to_global.assign(static_cast<size_t>(std::max(0, max_local_all) + 1), -1);

    // Build detections (may filter by area/feature coverage)
    std::vector<DIFDetection> dets = BuildDetections(label_map, depth, Tcw, camera, allow_3d_update, keypoints_uv);

    // Build track list index
    std::vector<int> track_ids;
    track_ids.reserve(mTracks.size());
    for(const auto& kv : mTracks)
        track_ids.push_back(kv.first);
    // Deterministic ordering: avoids unstable tie-breaking in Hungarian when multiple assignments share
    // the same minimum cost (unordered_map iteration order can change after rehash/erase/insert).
    std::sort(track_ids.begin(), track_ids.end());

    const int M = static_cast<int>(track_ids.size());
    const int N = static_cast<int>(dets.size());

    std::vector<int> matched_track_idx(M, -1);
    std::vector<int> matched_det_idx(N, -1);

    if(M > 0 && N > 0)
    {
        std::vector<std::vector<double>> C(static_cast<size_t>(M), std::vector<double>(static_cast<size_t>(N), 1.0));

        for(int mi = 0; mi < M; ++mi)
        {
            const DIFTrack& tr = mTracks.at(track_ids[mi]);
            for(int di = 0; di < N; ++di)
            {
                const DIFDetection& det = dets[di];

                const float s_iou = DIFMaskIoU(tr.mask_last, det.mask);
                const float d2d = cv::norm(tr.c2d_last - det.c2d);
                const float s2d = GaussianScore(d2d, mCfg.sigma_2d);

                const bool has3d = tr.has_Cc && tr.has_Tcw && det.valid3d;
                float d3d = 0.0f;
                float s3d = 0.0f;
                if(has3d)
                {
                    const Sophus::SE3f Trel = Tcw * tr.Tcw_last.inverse(); // c_last -> c_t
                    const Eigen::Vector3f Cc_pred = Trel * tr.Cc_last;
                    d3d = (Cc_pred - det.Cc).norm();
                    s3d = GaussianScore(d3d, mCfg.sigma_3d);
                }

                // Hard gating (reduce mismatch and Hungarian effective size).
                // For low-texture dynamic objects (e.g., balloons), 2D centroid can move substantially and IoU can still
                // be informative. Relax the 2D gate when IoU is non-trivial or 3D gate indicates compatibility.
                if(mCfg.tau_2d_max > 0.0f && d2d > mCfg.tau_2d_max)
                {
                    const float iou_relax = 0.05f;
                    bool allow = (s_iou >= iou_relax);
                    if(!allow && has3d)
                    {
                        float gate3d = mCfg.gate_3d_max;
                        if(!(gate3d > 0.0f))
                            gate3d = mCfg.v_max * static_cast<float>(std::max(0.0, timestamp - tr.last_timestamp)) + mCfg.margin_3d;
                        if(gate3d > 0.0f && d3d <= gate3d)
                            allow = true;
                    }
                    if(!allow)
                    {
                        C[static_cast<size_t>(mi)][static_cast<size_t>(di)] = kInfCost;
                        continue;
                    }
                }
                if(mCfg.tau_iou_min > 0.0f && s_iou < mCfg.tau_iou_min)
                {
                    C[static_cast<size_t>(mi)][static_cast<size_t>(di)] = kInfCost;
                    continue;
                }
                if(mCfg.bbox_area_ratio_max > 0.0f && tr.mask_last.area > 0 && det.area > 0)
                {
                    const float r = static_cast<float>(det.area) / static_cast<float>(std::max(1, tr.mask_last.area));
                    const float rr = std::max(r, 1.0f / std::max(r, 1e-6f));
                    if(rr > mCfg.bbox_area_ratio_max)
                    {
                        C[static_cast<size_t>(mi)][static_cast<size_t>(di)] = kInfCost;
                        continue;
                    }
                }
                if(has3d)
                {
                    float gate3d = mCfg.gate_3d_max;
                    if(!(gate3d > 0.0f))
                        gate3d = mCfg.v_max * static_cast<float>(std::max(0.0, timestamp - tr.last_timestamp)) + mCfg.margin_3d;
                    if(gate3d > 0.0f && d3d > gate3d)
                    {
                        C[static_cast<size_t>(mi)][static_cast<size_t>(di)] = kInfCost;
                        continue;
                    }
                }

                float w3d = has3d ? mCfg.w_3d : 0.0f;
                const float wsum = mCfg.w_iou + mCfg.w_2d + w3d;
                float score = 0.0f;
                if(wsum > 1e-6f)
                    score = (mCfg.w_iou * s_iou + mCfg.w_2d * s2d + w3d * s3d) / wsum;

                score = std::max(0.0f, std::min(1.0f, score));
                const double cost = 1.0 - static_cast<double>(score);
                C[static_cast<size_t>(mi)][static_cast<size_t>(di)] = std::min(cost, kInfCost);
            }
        }

        AssignmentResult asg = HungarianMinimize(C);
        for(int mi = 0; mi < M; ++mi)
        {
            const int di = (mi < static_cast<int>(asg.row_to_col.size())) ? asg.row_to_col[mi] : -1;
            if(di < 0 || di >= N)
                continue;
            const double cost = C[static_cast<size_t>(mi)][static_cast<size_t>(di)];
            if(cost >= kInfCost)
                continue;
            if(cost > static_cast<double>(mCfg.cost_accept))
                continue;
            matched_track_idx[mi] = di;
            matched_det_idx[di] = mi;
        }
    }

    // Update matched tracks
    for(int mi = 0; mi < M; ++mi)
    {
        const int di = matched_track_idx[mi];
        DIFTrack& tr = mTracks[track_ids[mi]];
        if(di < 0)
        {
            tr.miss_count++;
            continue;
        }

        const DIFDetection& det = dets[di];

        const bool had_Cc = tr.has_Cc;
        const Eigen::Vector3f prev_Cc = tr.Cc_last;
        const bool had_Tcw = tr.has_Tcw;
        const Sophus::SE3f prev_Tcw = tr.Tcw_last;
        const double prev_ts = tr.last_timestamp;
        const bool had_v = (tr.last_v_obs_frame >= 0);
        const float prev_v = tr.v_obs;
        const float match_iou = DIFMaskIoU(tr.mask_last, det.mask);

        tr.age++;
        tr.last_frame = frame_id;
        tr.miss_count = 0;
        tr.mature_count++;
        tr.bbox_last = det.bbox;
        tr.c2d_last = det.c2d;
        tr.mask_last = det.mask;

        // 3D observation (for association) and optional velocity observation (v_obs)
        tr.last_v_obs_frame = -1;
        if(det.valid3d)
        {
            if(mCfg.vobs_mode == 0)
            {
                const double dt = (prev_ts > 0.0) ? (timestamp - prev_ts) : 0.0;
                const bool dt_ok = (dt > 1e-3) && (mCfg.vobs_dt_min <= 0.0f || dt >= static_cast<double>(mCfg.vobs_dt_min));
                if(had_Cc && had_Tcw && dt_ok && match_iou >= mCfg.vobs_iou_min)
                {
                    const Sophus::SE3f Trel = Tcw * prev_Tcw.inverse(); // c_prev -> c_cur
                    const Eigen::Vector3f Cc_pred = Trel * prev_Cc;
                    const float delta = (det.Cc - Cc_pred).norm();

                    // Depth-aware noise-floor subtraction from pixel jitter.
                    float delta_bias = 0.0f;
                    if(mCfg.vobs_px_sigma > 0.0f && f_avg > 1e-6f && det.Cc.z() > 1e-6f)
                    {
                        const float z = std::max(0.0f, det.Cc.z());
                        delta_bias = std::sqrt(2.0f) * z * mCfg.vobs_px_sigma / f_avg;
                    }
                    const float delta_eff = std::max(0.0f, delta - delta_bias);
                    const float v_meas = delta_eff / std::max(static_cast<float>(dt), 1e-6f);

                    // Suppress v_obs for large planar/background-like instances (wall/floor/table-top), unless motion is extreme.
                    bool suppress_v = false;
                    if(mCfg.vobs_bg_suppress_enable && image_area > 0 && det.z_sigma_rob > 0.0f)
                    {
                        const float area_ratio = static_cast<float>(det.area) / static_cast<float>(image_area);
                        const bool bg_like =
                            (area_ratio >= mCfg.vobs_bg_area_ratio_min) && (det.z_sigma_rob <= mCfg.vobs_bg_z_sigma_max);
                        if(bg_like && v_meas < mCfg.vobs_bg_v_min)
                            suppress_v = true;
                    }
                    const float beta = std::max(0.0f, std::min(1.0f, mCfg.vel_ema_beta));
                    if(!suppress_v)
                    {
                        tr.v_obs = had_v ? ((1.0f - beta) * prev_v + beta * v_meas) : v_meas;
                        tr.v_obs_q = 1.0f;
                        tr.last_v_obs_frame = frame_id;
                    }
                }
            }

            tr.Cc_last = det.Cc;
            tr.has_Cc = true;
            tr.Tcw_last = Tcw;
            tr.has_Tcw = true;
        }

        tr.last_timestamp = timestamp;
    }

    // Build local->global mapping and create new tracks for unmatched detections
    for(int di = 0; di < N; ++di)
    {
        const int mi = matched_det_idx[di];
        const DIFDetection& det = dets[di];
        int gid = -1;
        if(mi >= 0)
        {
            gid = track_ids[mi];
        }
        else
        {
            gid = AllocateNewTrackId();
            DIFTrack tr;
            tr.id = gid;
            tr.age = 1;
            tr.last_frame = frame_id;
            tr.last_timestamp = timestamp;
            tr.miss_count = 0;
            tr.mature_count = 1;
            tr.bbox_last = det.bbox;
            tr.c2d_last = det.c2d;
            tr.mask_last = det.mask;
            if(det.valid3d)
            {
                tr.has_Cc = true;
                tr.Cc_last = det.Cc;
                tr.has_Tcw = true;
                tr.Tcw_last = Tcw;
            }
            mTracks[gid] = std::move(tr);
        }

        if(det.local_id >= 0 && det.local_id < static_cast<int>(out_local_to_global.size()))
            out_local_to_global[static_cast<size_t>(det.local_id)] = gid;
    }

    // Prune tracks with too many misses.
    std::vector<int> to_erase;
    for(const auto& kv : mTracks)
    {
        const DIFTrack& tr = kv.second;
        if(tr.miss_count > mCfg.max_miss)
            to_erase.push_back(kv.first);
    }
    for(const int id : to_erase)
        mTracks.erase(id);

    // NOTE: Velocity computation requires storing previous timestamp/centroid; this will be finalized when integrating frame caches in Tracking.
    return true;
}

} // namespace ORB_SLAM3
