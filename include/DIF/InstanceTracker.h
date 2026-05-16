#ifndef ORB_SLAM3_DIF_INSTANCE_TRACKER_H
#define ORB_SLAM3_DIF_INSTANCE_TRACKER_H

#include <limits>
#include <unordered_map>
#include <vector>

#include <opencv2/core/core.hpp>

#include <sophus/se3.hpp>

#include "DIF/InstanceTypes.h"

namespace ORB_SLAM3 {

class GeometricCamera;

struct DIFInstanceTrackerConfig
{
    bool enable = false;

    // Detection filtering
    int area_min = 200;
    int max_masks = 200;
    // Drop instances with too few ORB keypoints inside (0 disables).
    int n_feat_min = 0;
    // Shape-quality filtering (0 disables):
    // - bbox_min_w/h: reject tiny/thin instances
    // - fill_ratio_min: reject spiky edge fragments (area / bbox_area)
    int bbox_min_w = 0;
    int bbox_min_h = 0;
    float fill_ratio_min = 0.0f;

    // Pose-quality gate for 3D centroid update (Module B 8.2)
    // If pose is poor, skip 3D centroid/velocity update (valid3d=false), but still do 2D/IoU association.
    int pose_inliers_min = 50;
    float pose_rbg_max = std::numeric_limits<float>::infinity(); // px; <=0 disables; INF means ignore

    // 3D centroid (mask sampling)
    int sample_stride = 4;
    int sample_max_points = 3000;
    float depth_min = 0.2f;
    float depth_max = 6.0f;
    int n_in_min = 50;
    float mad_tau = 3.0f;

    // Matching cost (Module B)
    float w_iou = 0.4f;
    float w_2d = 0.3f;
    float w_3d = 0.3f;
    float sigma_2d = 40.0f;
    float sigma_3d = 0.6f;

    // Hard gating before Hungarian (Module B 3.4.5-C)
    float tau_2d_max = 150.0f;     // px
    float gate_3d_max = 1.5f;      // m; <=0 uses (v_max*dt + margin_3d) instead (legacy)
    float bbox_area_ratio_max = 4.0f; // max(area_ratio, 1/area_ratio) (<=0 disables)
    float tau_iou_min = 0.0f;      // optional; <=0 disables hard IoU gate
    float v_max = 2.0f;            // m/s (legacy 3D gate; only used when gate_3d_max<=0)
    float margin_3d = 0.5f;        // m   (legacy 3D gate; only used when gate_3d_max<=0)

    float cost_accept = 0.7f;
    int max_miss = 20;

    // Velocity smoothing
    float vel_ema_beta = 0.5f;

    // Velocity observation reliability gate:
    // Skip v_obs update when matched mask IoU is too low (segmentation jitter / association uncertainty).
    float vobs_iou_min = 0.20f;

    // v_obs mode:
    // - 0: centroid residual speed (legacy)
    // - 1: feature 3D residual speed (recommended; see docs/参考资料/DIF_SLAM_工程实现方案.md)
    int vobs_mode = 1;

    // v_obs robustness common params:
    // - vobs_px_sigma: expected pixel-level jitter (px), used to subtract a depth-aware motion noise floor.
    // - vobs_dt_min: minimum dt (seconds); when dt < vobs_dt_min, do not emit v_obs (avoid amplifying noise).
    float vobs_px_sigma = 2.0f; // px; <=0 disables pixel-jitter component
    float vobs_dt_min = 0.0f;   // seconds; <=0 disables dt gate

    // v_obs (feature 3D residual) params (Mode=1)
    int vobs_match_n_min = 15;          // minimum residual samples to update v_obs
    int vobs_match_n_max = 200;         // cap samples per instance for runtime
    float vobs_depth_jump_ratio_max = 0.2f; // |z_t - z_{t-1}| / max(z_{t-1}, eps) gate (<=0 disables)
    int vobs_mask_erode_px = 2;         // erosion radius in pixels (0 disables)
    float vobs_edge_ratio_min = 0.0f;   // require N_after_erode / N_before_erode >= this (<=0 disables)
    float vobs_mad_kappa = 2.5f;        // MAD inlier threshold multiplier
    float vobs_trim_frac = 0.2f;        // trimmed-mean fraction (0..0.49)
    float vobs_depth_sigma_m = 0.015f;  // depth noise sigma (meters) for bias floor (<=0 disables depth term)
    float vobs_sigma_r_max = 0.15f;     // max robust sigma of residuals (meters) for accepting v_obs (<=0 disables)
    float vobs_inlier_ratio_min = 0.5f; // require N_in / N_raw >= this (<=0 disables)
    float vobs_q_min = 0.2f;            // minimum quality to accept v_obs
    int vobs_q_n_ref = 30;              // quality reference sample count
    float vobs_q_sigma_ref = 0.05f;     // quality reference sigma (meters)
    bool vobs_bi_label_enable = false;  // optional: require gid consistency in both frames when prev label_map exists

    // Suppress speed observation on large planar/background-like instances (wall/floor/table-top), which tend to have
    // unstable centroids and can dominate false-D when using "Everything" segmentation.
    bool vobs_bg_suppress_enable = true;
    float vobs_bg_area_ratio_min = 0.15f; // >= area_ratio and planar -> treat as background-like
    float vobs_bg_z_sigma_max = 0.02f;    // meters (robust sigma on z)
    float vobs_bg_v_min = 2.0f;           // m/s; allow v_obs only if motion is extremely strong
};

class DIFInstanceTracker
{
public:
    explicit DIFInstanceTracker(const DIFInstanceTrackerConfig& cfg);

    void SetConfig(const DIFInstanceTrackerConfig& cfg);

    // Reset tracker state (tracks + id allocator). Useful when SLAM map is reset.
    void Reset();

    // Update tracks from a segmentation result (can be called at any time, independent of Tracking).
    // label_map: CV_16S, -1..N-1, mutually-exclusive.
    // depth: CV_32F in meters, aligned to label_map.
    // Tcw: camera pose for this frame (world -> camera).
    bool UpdateFromSegmentation(
        int frame_id,
        double timestamp,
        const cv::Mat& label_map,
        const cv::Mat& depth,
        const Sophus::SE3f& Tcw,
        GeometricCamera* camera,
        bool allow_3d_update,
        const std::vector<cv::Point2f>& keypoints_uv,
        std::vector<int>& out_local_to_global);

    const std::unordered_map<int, DIFTrack>& GetTracks() const { return mTracks; }
    std::unordered_map<int, DIFTrack>& GetTracksMutable() { return mTracks; }

    int AllocateNewTrackId();

private:
    DIFInstanceTrackerConfig mCfg;
    int mNextTrackId = 0;
    std::unordered_map<int, DIFTrack> mTracks;

private:
    std::vector<DIFDetection> BuildDetections(
        const cv::Mat& label_map,
        const cv::Mat& depth,
        const Sophus::SE3f& Tcw,
        GeometricCamera* camera,
        bool allow_3d_update,
        const std::vector<cv::Point2f>& keypoints_uv) const;

    static bool Compute3DCentroidMAD(
        const std::vector<Eigen::Vector3f>& points,
        int n_in_min,
        float mad_tau,
        Eigen::Vector3f& out_C,
        int* out_n_inliers,
        float* out_z_sigma_rob);

    static float GaussianScore(float d, float sigma);
};

} // namespace ORB_SLAM3

#endif
