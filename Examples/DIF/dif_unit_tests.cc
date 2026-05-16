#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "CameraModels/Pinhole.h"
#include "DIF/DynamicStateEstimator.h"
#include "DIF/InstanceTracker.h"
#include "Frame.h"

namespace {

void AssertTrue(bool cond, const char* msg)
{
    if(!cond)
    {
        std::cerr << "[FAIL] " << msg << std::endl;
        std::exit(1);
    }
}

void AssertNear(float a, float b, float eps, const char* msg)
{
    if(!(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= eps))
    {
        std::cerr << "[FAIL] " << msg << " a=" << a << " b=" << b << " eps=" << eps << std::endl;
        std::exit(1);
    }
}

// Reference linear-domain HMM update (speed-only), matching `DIFDynamicStateEstimator` implementation.
Eigen::Vector3f HmmUpdateLinear(
    const ORB_SLAM3::DIFDynamicStateConfig& cfg, const Eigen::Vector3f& alpha_prev, float v_obs)
{
    const float A00 = cfg.A[0], A01 = cfg.A[1], A02 = cfg.A[2];
    const float A10 = cfg.A[3], A11 = cfg.A[4], A12 = cfg.A[5];
    const float A20 = cfg.A[6], A21 = cfg.A[7], A22 = cfg.A[8];

    Eigen::Vector3f alpha_pred;
    alpha_pred.x() = alpha_prev.x() * A00 + alpha_prev.y() * A10 + alpha_prev.z() * A20;
    alpha_pred.y() = alpha_prev.x() * A01 + alpha_prev.y() * A11 + alpha_prev.z() * A21;
    alpha_pred.z() = alpha_prev.x() * A02 + alpha_prev.y() * A12 + alpha_prev.z() * A22;

    auto half_normal = [](float v, float sigma) -> float {
        if(!(sigma > 1e-6f) || v < 0.0f) return 0.0f;
        const float a = std::sqrt(2.0f / static_cast<float>(M_PI)) / sigma;
        const float x = v / sigma;
        return a * std::exp(-0.5f * x * x);
    };

    auto trunc_normal_at0 = [](float v, float mu, float sigma) -> float {
        if(!(sigma > 1e-6f) || v < 0.0f) return 0.0f;
        const float z = (v - mu) / sigma;
        const float phi = (1.0f / std::sqrt(2.0f * static_cast<float>(M_PI))) * std::exp(-0.5f * z * z);
        const float cdf = 0.5f * (1.0f + std::erf(mu / (sigma * std::sqrt(2.0f)))); // Phi(mu/sigma)
        const float denom = sigma * std::max(cdf, 1e-6f);
        return phi / denom;
    };

    const float v = std::min(std::max(v_obs, 0.0f), cfg.v_clip);
    const float pS = half_normal(v, cfg.sigma_S);
    const float pMS = half_normal(v, cfg.sigma_MS);
    const float pD = trunc_normal_at0(v, cfg.mu_D, cfg.sigma_D);

    Eigen::Vector3f alpha;
    alpha.x() = alpha_pred.x() * pS;
    alpha.y() = alpha_pred.y() * pMS;
    alpha.z() = alpha_pred.z() * pD;

    const float s = alpha.x() + alpha.y() + alpha.z();
    if(!(s > 1e-12f))
        return Eigen::Vector3f(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);
    alpha /= s;
    return alpha;
}

void TestModuleBWeightNormalizationNo3D()
{
    using namespace ORB_SLAM3;

    DIFInstanceTrackerConfig cfg;
    cfg.enable = true;
    cfg.area_min = 1;
    cfg.max_masks = 50;
    cfg.n_feat_min = 0;
    cfg.sample_stride = 1;
    cfg.sample_max_points = 1000;

    // Stress test for "missing-3D weight renormalization":
    // Set w_3d dominant but ensure association still succeeds when no 3D is available.
    cfg.w_iou = 0.10f;
    cfg.w_2d = 0.10f;
    cfg.w_3d = 0.80f;
    cfg.sigma_2d = 40.0f;
    cfg.sigma_3d = 0.6f;
    cfg.tau_2d_max = 1000.0f;
    cfg.gate_3d_max = 0.0f;
    cfg.cost_accept = 0.70f;

    DIFInstanceTracker tracker(cfg);

    const int H = 12;
    const int W = 16;
    cv::Mat label(H, W, CV_16S, cv::Scalar(-1));
    for(int y = 3; y <= 8; ++y)
        for(int x = 4; x <= 10; ++x)
            label.at<int16_t>(y, x) = 0;

    cv::Mat depth(H, W, CV_32F, cv::Scalar(1.0f));

    Pinhole cam(std::vector<float>{500.0f, 500.0f, 0.0f, 0.0f});
    const Sophus::SE3f Tcw(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());

    std::vector<int> l2g0;
    const bool ok0 = tracker.UpdateFromSegmentation(0, 0.0, label, depth, Tcw, &cam, /*allow_3d_update=*/false, {}, l2g0);
    AssertTrue(ok0, "UpdateFromSegmentation(0) must succeed");
    AssertTrue(tracker.GetTracks().size() == 1, "tracks size must be 1 after first update");
    AssertTrue(!l2g0.empty() && l2g0[0] == 0, "local->global[0] must be 0 after first update");

    std::vector<int> l2g1;
    const bool ok1 = tracker.UpdateFromSegmentation(1, 0.1, label, depth, Tcw, &cam, /*allow_3d_update=*/false, {}, l2g1);
    AssertTrue(ok1, "UpdateFromSegmentation(1) must succeed");
    AssertTrue(tracker.GetTracks().size() == 1, "tracks size must remain 1 (must match existing track)");
    AssertTrue(!l2g1.empty() && l2g1[0] == 0, "local->global[0] must remain 0 (must match existing track)");
}

void TestModuleBCentroidMADRobustness()
{
    using namespace ORB_SLAM3;

    DIFInstanceTrackerConfig cfg;
    cfg.enable = true;
    cfg.area_min = 1;
    cfg.max_masks = 50;
    cfg.n_feat_min = 0;
    cfg.sample_stride = 1;
    cfg.sample_max_points = 1000;
    cfg.depth_min = 0.1f;
    cfg.depth_max = 6.0f;
    cfg.n_in_min = 5;
    cfg.mad_tau = 3.0f;
    cfg.cost_accept = 1.0f; // irrelevant (single instance)

    DIFInstanceTracker tracker(cfg);

    const int H = 6;
    const int W = 6;
    cv::Mat label(H, W, CV_16S, cv::Scalar(-1));
    for(int y = 0; y < H; ++y)
        for(int x = 0; x < W; ++x)
            label.at<int16_t>(y, x) = 0;

    cv::Mat depth(H, W, CV_32F, cv::Scalar(1.0f));
    // Inject a few in-range outliers (will be rejected by Median+MAD on z).
    depth.at<float>(0, 0) = 3.0f;
    depth.at<float>(0, 1) = 3.0f;
    depth.at<float>(1, 0) = 3.0f;
    depth.at<float>(1, 1) = 3.0f;

    Pinhole cam(std::vector<float>{500.0f, 500.0f, 0.0f, 0.0f});
    const Sophus::SE3f Tcw(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());

    std::vector<int> l2g;
    const bool ok = tracker.UpdateFromSegmentation(0, 0.0, label, depth, Tcw, &cam, /*allow_3d_update=*/true, {}, l2g);
    AssertTrue(ok, "UpdateFromSegmentation must succeed");
    const auto& tracks = tracker.GetTracks();
    AssertTrue(tracks.size() == 1, "tracks size must be 1");

    const auto it = tracks.find(0);
    AssertTrue(it != tracks.end(), "track 0 must exist");
    const DIFTrack& tr = it->second;
    AssertTrue(tr.has_Cc, "track must have_Cc");
    AssertNear(tr.Cc_last.z(), 1.0f, 1e-3f, "robust centroid z must reject outliers");
}

void TestModuleCHMMLogDomainConsistency()
{
    using namespace ORB_SLAM3;

    DIFDynamicStateConfig cfg;
    cfg.enable = true;
    cfg.v_bg_enable = false;
    cfg.flow_enable = false;
    cfg.static_enable = false;
    cfg.mature_min = 0;
    cfg.mask_max_age_s = 1.0f;

    DIFDynamicStateEstimator est(cfg);

    Frame f;
    f.N = 0;
    f.mnId = 123;
    f.mTimeStamp = 10.0;

    std::unordered_map<int, DIFTrack> tracks;
    DIFTrack tr;
    tr.id = 0;
    tr.alpha = Eigen::Vector3f(0.20f, 0.50f, 0.30f);
    tr.last_v_obs_frame = static_cast<int>(f.mnId);
    tr.v_obs = 0.12f;
    tr.mature_count = 10;
    tracks.emplace(0, tr);

    std::unordered_map<int, std::vector<float>> inst_errors;
    (void)est.UpdateFromFrame(f, /*camera=*/nullptr, tracks, inst_errors, /*label_map=*/nullptr, /*local_to_global=*/nullptr,
                              /*flow_bg_errors=*/nullptr, /*flow_inst_errors=*/nullptr);

    const auto it = tracks.find(0);
    AssertTrue(it != tracks.end(), "track 0 must exist after update");
    const Eigen::Vector3f alpha_ref = HmmUpdateLinear(cfg, tr.alpha, tr.v_obs);
    const Eigen::Vector3f alpha_got = it->second.alpha;

    AssertNear(alpha_got.x(), alpha_ref.x(), 1e-5f, "alpha[S] mismatch");
    AssertNear(alpha_got.y(), alpha_ref.y(), 1e-5f, "alpha[MS] mismatch");
    AssertNear(alpha_got.z(), alpha_ref.z(), 1e-5f, "alpha[D] mismatch");
}

void TestModuleCHMMConsumesDelayedVObs()
{
    using namespace ORB_SLAM3;

    DIFDynamicStateConfig cfg;
    cfg.enable = true;
    cfg.v_bg_enable = false;
    cfg.flow_enable = false;
    cfg.static_enable = false;
    cfg.mature_min = 0;
    cfg.mask_max_age_s = 1.0f;

    DIFDynamicStateEstimator est(cfg);

    Frame f;
    f.N = 0;
    f.mnId = 200;
    f.mTimeStamp = 20.0;

    std::unordered_map<int, DIFTrack> tracks;
    DIFTrack tr;
    tr.id = 0;
    tr.alpha = Eigen::Vector3f(0.80f, 0.20f, 0.00f);
    tr.last_v_obs_frame = 190; // v_obs came from an older frame (async segmentation)
    tr.last_v_obs_used_frame = -1;
    tr.last_timestamp = 19.9; // dt=0.1s < mask_max_age_s
    tr.v_obs = 0.30f;
    tr.mature_count = 10;
    tracks.emplace(0, tr);

    std::unordered_map<int, std::vector<float>> inst_errors;
    (void)est.UpdateFromFrame(f, /*camera=*/nullptr, tracks, inst_errors, /*label_map=*/nullptr, /*local_to_global=*/nullptr,
                              /*flow_bg_errors=*/nullptr, /*flow_inst_errors=*/nullptr);

    const auto it = tracks.find(0);
    AssertTrue(it != tracks.end(), "track 0 must exist after delayed-vobs update");
    AssertTrue(it->second.last_v_obs_used_frame == 190, "delayed v_obs must be consumed exactly once");
}

void TestModuleCEnterDKRequiresTwoSpeedUpdates()
{
    using namespace ORB_SLAM3;

    DIFDynamicStateConfig cfg;
    cfg.enable = true;
    cfg.v_bg_enable = false;
    cfg.flow_enable = false;
    cfg.static_enable = false;
    cfg.mature_min = 0;
    cfg.enter_D_K = 2;
    cfg.tau_up = 0.70f;
    cfg.tau_down = 0.40f;

    DIFDynamicStateEstimator est(cfg);

    std::unordered_map<int, DIFTrack> tracks;
    DIFTrack tr;
    tr.id = 0;
    tr.alpha = Eigen::Vector3f(0.80f, 0.20f, 0.00f);
    tr.mature_count = 10;

    // First speed observation (v around mu_D makes alpha[D] dominate strongly).
    {
        Frame f;
        f.N = 0;
        f.mnId = 1;
        f.mTimeStamp = 1.0;

        tr.last_v_obs_frame = 1;
        tr.last_v_obs_used_frame = -1;
        tr.last_timestamp = 1.0;
        tr.v_obs = cfg.mu_D;

        tracks.clear();
        tracks.emplace(0, tr);

        std::unordered_map<int, std::vector<float>> inst_errors;
        (void)est.UpdateFromFrame(f, /*camera=*/nullptr, tracks, inst_errors, /*label_map=*/nullptr, /*local_to_global=*/nullptr,
                                  /*flow_bg_errors=*/nullptr, /*flow_inst_errors=*/nullptr);

        const DIFTrack& out = tracks.at(0);
        AssertTrue(out.state_hat != DIFTrackState::D, "enter_D_K=2: first v_obs must not enter D");
        AssertTrue(out.enter_dyn_count == 1, "enter_dyn_count must be 1 after first confirmation");
    }

    // Second speed observation should enter D.
    {
        Frame f;
        f.N = 0;
        f.mnId = 2;
        f.mTimeStamp = 2.0;

        DIFTrack in = tracks.at(0);
        in.last_v_obs_frame = 2;
        in.v_obs = cfg.mu_D;
        in.last_timestamp = 2.0;

        tracks.clear();
        tracks.emplace(0, in);

        std::unordered_map<int, std::vector<float>> inst_errors;
        (void)est.UpdateFromFrame(f, /*camera=*/nullptr, tracks, inst_errors, /*label_map=*/nullptr, /*local_to_global=*/nullptr,
                                  /*flow_bg_errors=*/nullptr, /*flow_inst_errors=*/nullptr);

        const DIFTrack& out = tracks.at(0);
        AssertTrue(out.state_hat == DIFTrackState::D, "enter_D_K=2: second v_obs must enter D");
        AssertTrue(out.enter_dyn_count == 0, "enter_dyn_count must reset after entering D");
    }
}

void TestModuleCStaticVLockForcesS()
{
    using namespace ORB_SLAM3;

    DIFDynamicStateConfig cfg;
    cfg.enable = true;
    cfg.v_bg_enable = false;
    cfg.flow_enable = false;
    cfg.static_enable = true;
    cfg.static_K = 3;
    cfg.mature_min = 0;
    cfg.enter_D_K = 3;

    DIFDynamicStateEstimator est(cfg);

    std::unordered_map<int, DIFTrack> tracks;
    DIFTrack tr;
    tr.id = 0;
    tr.alpha = Eigen::Vector3f(0.60f, 0.40f, 0.00f);
    tr.mature_count = 10;

    // Apply several tiny v_obs updates; should lock to S.
    for(int k = 0; k < 4; ++k)
    {
        Frame f;
        f.N = 0;
        f.mnId = 100 + k;
        f.mTimeStamp = 1.0 + 0.1 * k;

        tr.last_v_obs_frame = static_cast<int>(f.mnId);
        tr.last_v_obs_used_frame = -1;
        tr.last_timestamp = f.mTimeStamp;
        tr.v_obs = 0.0f; // static

        tracks.clear();
        tracks.emplace(0, tr);

        std::unordered_map<int, std::vector<float>> inst_errors;
        (void)est.UpdateFromFrame(f, /*camera=*/nullptr, tracks, inst_errors, /*label_map=*/nullptr, /*local_to_global=*/nullptr,
                                  /*flow_bg_errors=*/nullptr, /*flow_inst_errors=*/nullptr);

        const DIFTrack& out = tracks.at(0);
        tr = out; // carry state
    }

    const DIFTrack& out = tracks.at(0);
    AssertTrue(out.static_v_count >= cfg.static_K, "static_v_count must reach static_K");
    AssertTrue(out.state_hat == DIFTrackState::S, "static_v_lock must force S for non-ever_dynamic tracks");
}

} // namespace

int main()
{
    TestModuleBWeightNormalizationNo3D();
    TestModuleBCentroidMADRobustness();
    TestModuleCHMMLogDomainConsistency();
    TestModuleCHMMConsumesDelayedVObs();
    TestModuleCEnterDKRequiresTwoSpeedUpdates();
    TestModuleCStaticVLockForcesS();

    std::cout << "OK" << std::endl;
    return 0;
}
