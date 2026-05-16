#ifndef ORB_SLAM3_DIF_INSTANCE_TYPES_H
#define ORB_SLAM3_DIF_INSTANCE_TYPES_H

#include <cstdint>
#include <limits>
#include <vector>

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

struct DIFBbox
{
    int x1 = 0;
    int y1 = 0;
    int x2 = -1;
    int y2 = -1;

    bool IsValid() const { return x2 >= x1 && y2 >= y1; }
    int Width() const { return IsValid() ? (x2 - x1 + 1) : 0; }
    int Height() const { return IsValid() ? (y2 - y1 + 1) : 0; }
    int Area() const { return Width() * Height(); }
};

inline DIFBbox DIFIntersectBbox(const DIFBbox& a, const DIFBbox& b)
{
    DIFBbox out;
    out.x1 = std::max(a.x1, b.x1);
    out.y1 = std::max(a.y1, b.y1);
    out.x2 = std::min(a.x2, b.x2);
    out.y2 = std::min(a.y2, b.y2);
    if(!out.IsValid())
        return DIFBbox();
    return out;
}

inline DIFBbox DIFUnionBbox(const DIFBbox& a, const DIFBbox& b)
{
    if(!a.IsValid()) return b;
    if(!b.IsValid()) return a;
    DIFBbox out;
    out.x1 = std::min(a.x1, b.x1);
    out.y1 = std::min(a.y1, b.y1);
    out.x2 = std::max(a.x2, b.x2);
    out.y2 = std::max(a.y2, b.y2);
    return out;
}

struct DIFMaskRoi
{
    DIFBbox bbox;
    cv::Mat mask; // CV_8U, size == bbox (H,W), values {0,1} or {0,255}
    int area = 0; // count of non-zero pixels in mask

    bool Empty() const { return !bbox.IsValid() || mask.empty() || area <= 0; }
};

inline float DIFMaskIoU(const DIFMaskRoi& a, const DIFMaskRoi& b)
{
    if(a.Empty() || b.Empty())
        return 0.0f;

    const DIFBbox inter = DIFIntersectBbox(a.bbox, b.bbox);
    if(!inter.IsValid())
        return 0.0f;

    const int ax = inter.x1 - a.bbox.x1;
    const int ay = inter.y1 - a.bbox.y1;
    const int bx = inter.x1 - b.bbox.x1;
    const int by = inter.y1 - b.bbox.y1;
    const int w = inter.Width();
    const int h = inter.Height();

    const cv::Rect ra(ax, ay, w, h);
    const cv::Rect rb(bx, by, w, h);
    if(ra.x < 0 || ra.y < 0 || ra.x + ra.width > a.mask.cols || ra.y + ra.height > a.mask.rows)
        return 0.0f;
    if(rb.x < 0 || rb.y < 0 || rb.x + rb.width > b.mask.cols || rb.y + rb.height > b.mask.rows)
        return 0.0f;

    const cv::Mat ma = a.mask(ra);
    const cv::Mat mb = b.mask(rb);

    cv::Mat aand;
    cv::bitwise_and(ma, mb, aand);
    const int inter_area = cv::countNonZero(aand);
    const int union_area = a.area + b.area - inter_area;
    if(union_area <= 0)
        return 0.0f;
    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

struct DIFDetection
{
    int local_id = -1;
    DIFBbox bbox;
    int area = 0;
    cv::Point2f c2d{0.f, 0.f};

    bool valid3d = false;
    Eigen::Vector3f Cc = Eigen::Vector3f::Zero();
    int n_inliers_3d = 0;
    float z_sigma_rob = 0.0f; // robust sigma on z (meters), from Median+MAD

    DIFMaskRoi mask;
};

enum class DIFTrackState : int
{
    S = 0,
    MS = 1,
    D = 2
};

struct DIFTrack
{
    int id = -1;
    int age = 0;
    int last_frame = -1;
    double last_timestamp = -1.0;
    int miss_count = 0;
    int mature_count = 0;

    DIFBbox bbox_last;
    cv::Point2f c2d_last{0.f, 0.f};
    DIFMaskRoi mask_last;

    // Camera-frame 3D centroid (Module B): Cc in the camera coordinate of the observation frame.
    bool has_Cc = false;
    Eigen::Vector3f Cc_last = Eigen::Vector3f::Zero();

    // Pose at last observation frame (world -> camera). Used to compute relative pose compensation.
    bool has_Tcw = false;
    Sophus::SE3f Tcw_last;

    // Predicted 2D centroid at current pose/time (for optional ROI-mask warp).
    cv::Point2f u2d_pred{0.f, 0.f};

    // Module C state
    // HMM probability vector:
    // - legacy mode (3-state): [P(S), P(MS), P(D)]
    // - two-state mode: [P(S), 0, P(D)] (MS dim is unused and kept as 0 for backward compatibility)
    Eigen::Vector3f alpha = Eigen::Vector3f(0.80f, 0.20f, 0.00f);
    DIFTrackState state_hat = DIFTrackState::S;
    bool ever_dynamic = false;
    // DIF-SLAM v2.0: map_lock is a *lock period* indicator (not a hard gate).
    // - It must NOT affect pose optimization (Tracking).
    // - It must NOT forbid mapping (LocalMapping/DenseMapping) by itself.
    // - It is used to compute the lock weight w_lock and to manage rollback semantics.
    bool map_lock = true;
    int map_S_streak = 0;
    int map_cooldown = 0;
    // Lock weight in [w_lock_min, 1], computed from map_S_streak / N_confirm.
    // When map_lock == false, w_lock must be 1.
    float w_lock = 1.0f;
    int err_count = 0;
    int flow_err_count = 0;
    int static_err_count = 0;
    int static_flow_count = 0;
    int static_v_count = 0;

    // Speed observation (updated only on segmentation frames where we can compute 3D centroids).
    int last_v_obs_frame = -1;
    // The last v_obs frame that has been consumed by Module C (HMM emission update).
    // This enables asynchronous segmentation: v_obs may arrive late, and should still be applied exactly once.
    int last_v_obs_used_frame = -1;
    float v_obs = 0.0f;
    // Quality of the latest v_obs in [0,1]. Used for soft HMM emission update.
    float v_obs_q = 1.0f;

    // Consecutive "enter dynamic" confirmations (Module C). This is used to reduce false positives caused by a single
    // noisy v_obs update or transient association jitter; see DIFDynamicStateConfig::enter_D_K.
    int enter_dyn_count = 0;
};

} // namespace ORB_SLAM3

#endif
