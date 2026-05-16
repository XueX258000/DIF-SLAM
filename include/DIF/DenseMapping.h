#ifndef ORB_SLAM3_DIF_DENSE_MAPPING_H
#define ORB_SLAM3_DIF_DENSE_MAPPING_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "DIF/InstanceTypes.h"

namespace ORB_SLAM3 {

class KeyFrame;

struct DenseMappingConfig
{
    bool enable = false;
    std::string integrate_on = "keyframe"; // "keyframe" or "segframe"
    float voxel_size_m = 0.03f;
    int stride = 2;
    float depth_min = 0.3f;
    float depth_max = 6.0f;
    // Optional depth denoise for suppressing flying pixels / ghosting near depth discontinuities.
    // 0=off; if enabled must be odd (>=3), e.g. 3 or 5.
    int depth_median_ksize = 0;
    // Optional depth discontinuity rejection (0=off):
    // skip pixels whose local (3x3) depth range exceeds max(delta_m, delta_ratio * z).
    float depth_edge_delta_m = 0.0f;
    float depth_edge_delta_ratio = 0.0f;
    int depth_edge_min_neighbors = 4; // require enough valid depth samples to activate edge rejection
    int forbid_dilate_px = 3;
    // Fill the bbox of forbidden instances (D) in addition to their masks.
    // This is a conservative option to eliminate holes/under-segmentation leakage into background fusion.
    bool forbid_fill_bbox = false;
    int forbid_bbox_margin_px = 0;
    // Legacy/ablation (pre v2.0): minimum Stable-S frames before fusing an instance.
    // DIF-SLAM v2.0 requires that non-D instances are *allowed* to fuse regardless of map_lock, so this should
    // normally be kept at 1.
    int n_dense_confirm = 1;
    bool require_exact_label_map = true;
    int queue_max = 10;
    bool rebuild_after_loop = false;
    // DIF-SLAM v2.0 rollback (de-fusion) support.
    bool defusion_enable = true;
    // Instance boundary safety margin (pixels). DenseMapping skips instance pixels that are too close to any label
    // boundary (background<->instance or instance<->instance) to suppress boundary jitter ghosting.
    int instance_guard_px = 2;
    int min_voxel_weight = 1;
    // Optional: require a voxel to be observed in at least N distinct frames (keyframes/segframes).
    // This suppresses single-frame depth artifacts / pose outliers that can look like "ghosting".
    // 0=off.
    int min_voxel_frames = 0;
    // Optional voxel-neighborhood outlier filter at export time (0=off).
    // Requires at least this many occupied 26-neighborhood voxels (after min_voxel_weight).
    int min_voxel_neighbors = 0;
};

enum class DensePacketKind : int
{
    Frame = 0,
    TrackEnterD = 1
};

struct DenseTrackStateInfo
{
    DIFTrackState state = DIFTrackState::S;
    bool map_lock = false;
    float w_lock = 1.0f;
};

struct DenseKFPacket
{
    DensePacketKind kind = DensePacketKind::Frame;
    int event_track_id = -1; // for kind==TrackEnterD

    int kf_id = -1;
    int frame_id = -1;
    double timestamp = -1.0;
    Sophus::SE3f Tcw;
    bool has_pose = false;

    cv::Mat rgb;       // CV_8UC3 (BGR or RGB, see is_rgb)
    cv::Mat depth;     // CV_32F, meters
    cv::Mat label_map; // CV_16S, value range [-1, N-1]
    cv::Mat forbid_mask; // CV_8U, optional extra forbid mask (D pixels only)
    std::vector<int> local_to_global;
    std::unordered_map<int, DenseTrackStateInfo> track_states;

    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    bool is_rgb = false;

    KeyFrame* pKF = nullptr;
};

class DenseMapping
{
public:
    explicit DenseMapping(const DenseMappingConfig& cfg);
    ~DenseMapping();

    void SetConfig(const DenseMappingConfig& cfg);

    void Start();
    void Stop();
    void Reset();

    void Enqueue(DenseKFPacket&& packet);
    void EnqueueTrackEnterD(int track_id, int frame_id, double timestamp);

    bool SaveDensePointCloud(const std::string& path, bool use_latest_pose = true);
    void RebuildGlobalVoxels(bool use_latest_pose = true);

    bool IsRunning() const { return mbRunning; }

private:
    struct DensePoint
    {
        Eigen::Vector3f pc = Eigen::Vector3f::Zero();
        Eigen::Vector3f rgb = Eigen::Vector3f::Zero(); // RGB in [0,255]
        int instance_id = -1;
        float w_lock = 1.0f;
        uint32_t gen = 0;
    };

    struct DenseKeyFrameCloud
    {
        int kf_id = -1;
        int frame_id = -1;
        double timestamp = -1.0;
        KeyFrame* pKF = nullptr;
        Sophus::SE3f Tcw;
        std::vector<DensePoint> points;
    };

    struct VoxelKey
    {
        int x = 0;
        int y = 0;
        int z = 0;
        VoxelKey() = default;
        VoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
        bool operator==(const VoxelKey& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash
    {
        size_t operator()(const VoxelKey& k) const
        {
            const size_t h1 = std::hash<int>{}(k.x);
            const size_t h2 = std::hash<int>{}(k.y);
            const size_t h3 = std::hash<int>{}(k.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    struct TrackAccum
    {
        int gid = -1;
        Eigen::Vector3f sum_p = Eigen::Vector3f::Zero();
        Eigen::Vector3f sum_rgb = Eigen::Vector3f::Zero();
        float w = 0.0f;
    };

    struct VoxelAccum
    {
        Eigen::Vector3f sum_p = Eigen::Vector3f::Zero();
        Eigen::Vector3f sum_rgb = Eigen::Vector3f::Zero();
        float w_total = 0.0f;
        int obs_frames = 0;
        int last_obs_frame_id = std::numeric_limits<int>::min();
        std::vector<TrackAccum> contrib; // per-track accumulations for rollback
    };

private:
    void ThreadMain();
    void ProcessPacket(DenseKFPacket& packet);
    void DefuseTrack(int track_id);
    cv::Mat BuildForbidMask(const DenseKFPacket& packet) const;
    cv::Mat BuildUnsafeMaskFromLabelMap(const cv::Mat& label_map) const;
    bool ConvertRgbToBgrIfNeeded(const DenseKFPacket& packet, cv::Mat& out_bgr) const;
    void IntegratePoint(const Eigen::Vector3f& pw,
                        const Eigen::Vector3f& rgb,
                        int instance_id,
                        float w_lock,
                        uint32_t gen,
                        int frame_id);

    bool ExportVoxelsToPLY(const std::string& path,
                           const std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash>& voxels) const;

    void BuildVoxelMapFromClouds(
        bool use_latest_pose,
        std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash>& out_voxels,
        std::unordered_map<int, std::unordered_set<VoxelKey, VoxelKeyHash>>& out_track_to_voxels) const;

private:
    DenseMappingConfig mCfg;

    std::atomic<bool> mbStop{false};
    std::atomic<bool> mbRunning{false};
    std::thread mThread;

    std::mutex mMutexQueue;
    std::condition_variable mCvQueue;
    std::deque<DenseKFPacket> mQueue;

    mutable std::mutex mMutexData;
    std::unordered_map<int, DenseKeyFrameCloud> mKeyFrameClouds;
    std::unordered_set<int> mIntegratedKFIds;
    std::vector<DenseKeyFrameCloud> mSegFrameClouds;

    std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> mGlobalVoxels;
    std::unordered_map<int, std::unordered_set<VoxelKey, VoxelKeyHash>> mTrackToVoxels;
    std::unordered_map<int, uint32_t> mTrackGeneration;
};

} // namespace ORB_SLAM3

#endif // ORB_SLAM3_DIF_DENSE_MAPPING_H
