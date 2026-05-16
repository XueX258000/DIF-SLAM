#ifndef ORB_SLAM3_DIF_DYNAMIC_STATE_ESTIMATOR_H
#define ORB_SLAM3_DIF_DYNAMIC_STATE_ESTIMATOR_H

#include <unordered_map>
#include <vector>

#include <opencv2/core/core.hpp>

#include "DIF/InstanceTypes.h"

namespace ORB_SLAM3 {

class Frame;
class GeometricCamera;
class MapPoint;

struct DIFDynamicStateConfig
{
    bool enable = false;

    // State mode:
    // - 0: legacy 3-state HMM {S, MS, D}
    // - 1: DIF-SLAM v2.0 2-state HMM {S, D} + map_lock (lock period) + w_lock
    int state_mode = 0;

    // Master switch for state-based suppression in ApplyDIFMaskToMatchedMapPoints (controls suppressed track_id rejection).
    // If false, only mask-based rejection is used; state-based rejection via suppressed track IDs is disabled.
    bool suppress_enable = true;

    // Legacy/ablation (pre v2.0): once a track has ever been dynamic, optionally suppress/clean its bound MapPoints.
    // DIF-SLAM v2.0 uses rollback (on nonD->D) instead of "ever_dynamic" permanent suppression.
    bool suppress_ever_dynamic = false;
    bool prune_ever_dynamic = false;

    // HMM transition matrix A (row-major: from {S,MS,D} to {S,MS,D})
    float A[9] = {
        0.97f, 0.03f, 0.00f,
        0.05f, 0.90f, 0.05f,
        0.00f, 0.15f, 0.85f
    };

    // 2-state HMM transition matrix A2 (row-major: from {S,D} to {S,D}).
    // Default values are from docs/参考资料/DIF_SLAM_工程实现方案.md.
    float A2[4] = {
        0.97f, 0.03f,
        0.10f, 0.90f
    };

    // Emission params (speed v in m/s)
    float sigma_S = 0.02f;
    float sigma_MS = 0.05f;
    float mu_D = 0.20f;
    float sigma_D = 0.10f;
    float v_clip = 3.0f;

    // DIF-SLAM v2.0 map_lock (state_mode=1):
    // - map_lock indicates a lock period (uncertain-static) and must NOT hard-forbid mapping.
    // - map_lock must NOT affect pose optimization (Tracking).
    // - map_lock drives the lock-weight schedule w_lock (used by sparse BA residuals and dense fusion).
    bool map_lock_enable = true;
    int map_confirm_N = 5;   // N_map_confirm (>=1)
    int map_cooldown_N = 5;  // N_map_cooldown (>=0)
    float w_lock_min = 0.2f; // w_min in w_lock (>=0, <=1)

    // Build M_t^{dyn} from tracks:
    // - false: only D pixels (DIF-SLAM v2.0 required semantics)
    // - true: D ∪ ever_dynamic (legacy/ablation)
    bool dyn_mask_include_ever_dynamic = false;

    // Speed observation bias removal: subtract a robust "background" speed level (computed from the smallest fraction
    // of per-track v_obs on the current frame) to reduce false positives caused by pose/segmentation jitter.
    bool v_bg_enable = true;
    int v_bg_n_min = 20;
    float v_bg_trim_frac = 0.60f;

    // Hysteresis thresholds on alpha[D]
    float tau_up = 0.70f;
    float tau_down = 0.40f;

    // Cold-start maturity gate (Module C 3.5.6)
    int mature_min = 3;

    // Circuit breaker thresholds (pixel domain)
    int n_min = 30;
    int n_bg_min = 80;
    // Background baseline fallback:
    // When explicit background (mnInstanceId == -1) residuals are insufficient (common with "Everything" segmentation),
    // estimate r_bg from the median of the smallest fraction of all inlier residuals.
    bool r_bg_fallback_enable = true;
    float r_bg_trim_frac = 0.60f;
    float tau_bg = 2.0f;
    float tau_q = 2.5f;
    float tau_dr = 1.5f;
    // Absolute reprojection error floor (px) for breaker activation (<=0 disables).
    // This mitigates false positives when r_bg is very small and ratio tests become overly sensitive.
    float tau_r_m_min = 2.5f;
    int K = 2;
    float lambda_D = 1.0f;
    float lambda_other = 1e-3f;
    float eps = 1e-6f;

    // Frontend rigid-flow residual circuit breaker (Module C 3.5.5-B)
    // Rigid-flow residual computation mode:
    // - 0: "desc" (default): ORB descriptor matching -> residual e=||u_cur - u_hat|| in px
    // - 1: "lk": LK optical flow (with u_hat as initial guess) -> residual e=||u_lk - u_hat|| in px
    int flow_mode = 0;
    bool flow_enable = true;
    int flow_n_min = 15;
    int flow_n_bg_min = 80;
    // Similar to r_bg_fallback_enable, but for flow baseline e_bg.
    // If flow_bg has too few samples (e.g., almost every keypoint belongs to some instance),
    // estimate e_bg from the median of the smallest fraction of all flow errors.
    bool flow_bg_fallback_enable = true;
    float flow_tau_bg = 1.5f;
    float flow_tau_q = 2.5f;
    float flow_tau_de = 1.0f;
    // Absolute flow error floor (px) for flow-breaker activation (<=0 disables).
    float flow_tau_e_m_min = 1.0f;
    int flow_K = 2;
    float flow_bg_trim_frac = 0.60f; // median of smallest fraction to reduce dynamic contamination

    // Static evidence suppression (reduce false positives on static background)
    bool static_enable = true;
    int static_K = 3;
    float static_tau_q = 1.30f;
    float static_tau_dr = 0.50f; // px
    float static_flow_tau_q = 1.30f;
    float static_flow_tau_de = 0.50f; // px
    float static_lambda_S = 1.05f;
    float static_lambda_MS = 1.00f;
    float static_lambda_D = 0.30f;

    // When static evidence suppression is on (static_*_count >= static_K), use stronger hysteresis to avoid entering D
    // due to noisy speed observations and to recover faster from false D.
    float static_hyst_tau_up = 0.85f;
    float static_hyst_tau_down = 0.60f;

    // Enter-D confirmation: require K consecutive "enter" confirmations (speed-driven, non-breaker) before switching
    // to state D. Breaker (reprojection/flow) still enters D immediately.
    int enter_D_K = 2;

    // Dynamic mask post-process
    int mask_dilate = 0;
    // Additional dilation growth per mask staleness (frames). 0 disables.
    // This is a conservative mitigation for asynchronous/low-frequency segmentation (E2).
    int mask_dilate_per_lag = 0;

    // Mask freshness gate (mitigate segmentation lag):
    // If > 0, exclude a track from M_t^{dyn} when (t_now - track.last_timestamp) exceeds this threshold.
    // This helps avoid applying very stale instance masks.
    float mask_max_age_s = 0.0f;

    // Lightweight mask propagation (translation warp):
    // If enabled, shift each dynamic track's last ROI mask based on the predicted 2D centroid at current pose/time.
    // This is a low-cost mitigation for asynchronous/low-frequency segmentation alignment (Issue #1).
    bool mask_warp_enable = false;
    float mask_warp_max_dt = 1.0f;   // seconds; skip warp if last obs too old
    float mask_warp_max_px = 200.0f; // pixels; skip warp if shift is too large
};

class DIFDynamicStateEstimator
{
public:
    explicit DIFDynamicStateEstimator(const DIFDynamicStateConfig& cfg);

    void SetConfig(const DIFDynamicStateConfig& cfg);

    // Update tracks' alpha/state_hat/err_count based on current frame inlier residuals and per-track speed obs.
    // Returns r_bg (median background reprojection error) for logging; INF if background not available.
    float UpdateFromFrame(
        const Frame& frame,
        GeometricCamera* camera,
        std::unordered_map<int, DIFTrack>& tracks,
        std::unordered_map<int, std::vector<float>>& out_inst_errors,
        const cv::Mat* label_map,
        const std::vector<int>* local_to_global,
        const std::vector<float>* flow_bg_errors,
        const std::unordered_map<int, std::vector<float>>* flow_inst_errors);

    // Build M_t^{dyn} from tracks' last masks (or current), using state_hat == D.
    cv::Mat BuildDynamicMask(int height, int width, const std::unordered_map<int, DIFTrack>& tracks) const;

private:
    DIFDynamicStateConfig mCfg;

private:
    static float HalfNormalPdf(float v, float sigma);
    static float TruncatedNormalPdfAt0(float v, float mu, float sigma);
    static float Median(std::vector<float>& values);
    static float MedianOfSmallestFraction(std::vector<float>& values, float frac);
    static cv::Point2f Project(const GeometricCamera* cam, const Eigen::Vector3f& Xc, bool& ok);
    static float ReprojectionErrorPx(const Frame& frame, GeometricCamera* camera, const MapPoint* mp, const cv::KeyPoint& kp);
    static void Normalize(Eigen::Vector3f& p);
};

} // namespace ORB_SLAM3

#endif
