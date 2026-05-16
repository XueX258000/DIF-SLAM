#include "DIF/DynamicStateEstimator.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

#include "CameraModels/GeometricCamera.h"
#include "Frame.h"
#include "MapPoint.h"

namespace ORB_SLAM3 {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

float LogSumExp3(const Eigen::Vector3f& x)
{
    const float m = std::max(x.x(), std::max(x.y(), x.z()));
    if(!std::isfinite(m))
        return m;
    return m + std::log(std::exp(x.x() - m) + std::exp(x.y() - m) + std::exp(x.z() - m));
}

float LogSumExp2(float a, float b)
{
    const float m = std::max(a, b);
    if(!std::isfinite(m))
        return m;
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

float SafeLog(float x)
{
    return std::log(std::max(x, 1e-12f));
}

} // namespace

DIFDynamicStateEstimator::DIFDynamicStateEstimator(const DIFDynamicStateConfig& cfg) : mCfg(cfg) {}

void DIFDynamicStateEstimator::SetConfig(const DIFDynamicStateConfig& cfg) { mCfg = cfg; }

float DIFDynamicStateEstimator::HalfNormalPdf(float v, float sigma)
{
    if(!(sigma > 1e-6f))
        return 0.0f;
    if(v < 0.0f)
        return 0.0f;
    const float a = std::sqrt(2.0f / static_cast<float>(M_PI)) / sigma;
    const float x = v / sigma;
    return a * std::exp(-0.5f * x * x);
}

float DIFDynamicStateEstimator::TruncatedNormalPdfAt0(float v, float mu, float sigma)
{
    if(!(sigma > 1e-6f))
        return 0.0f;
    if(v < 0.0f)
        return 0.0f;
    const float z = (v - mu) / sigma;
    const float phi = (1.0f / std::sqrt(2.0f * static_cast<float>(M_PI))) * std::exp(-0.5f * z * z);
    const float cdf = 0.5f * (1.0f + std::erf(mu / (sigma * std::sqrt(2.0f)))); // Phi(mu/sigma)
    const float denom = sigma * std::max(cdf, 1e-6f);
    return phi / denom;
}

float DIFDynamicStateEstimator::Median(std::vector<float>& values)
{
    if(values.empty())
        return kInf;
    const size_t k = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + k, values.end());
    return values[k];
}

float DIFDynamicStateEstimator::MedianOfSmallestFraction(std::vector<float>& values, float frac)
{
    if(values.empty())
        return kInf;
    frac = std::max(0.0f, std::min(1.0f, frac));
    if(frac <= 0.0f)
        return kInf;
    const size_t n = values.size();
    const size_t keep = std::max<size_t>(1, static_cast<size_t>(std::lround(frac * static_cast<float>(n))));
    std::nth_element(values.begin(), values.begin() + static_cast<long>(keep) - 1, values.end());
    values.resize(keep);
    return Median(values);
}

cv::Point2f DIFDynamicStateEstimator::Project(const GeometricCamera* cam, const Eigen::Vector3f& Xc, bool& ok)
{
    ok = false;
    if(!cam)
        return {};
    if(Xc.z() <= 1e-6f)
        return {};
    const Eigen::Vector2f uv = const_cast<GeometricCamera*>(cam)->project(Xc);
    ok = std::isfinite(uv.x()) && std::isfinite(uv.y());
    return cv::Point2f(uv.x(), uv.y());
}

float DIFDynamicStateEstimator::ReprojectionErrorPx(const Frame& frame, GeometricCamera* camera, const MapPoint* mp, const cv::KeyPoint& kp)
{
    if(!mp || !camera)
        return kInf;
    const Eigen::Vector3f Xw = const_cast<MapPoint*>(mp)->GetWorldPos();
    const Eigen::Vector3f Xc = frame.GetPose() * Xw;
    bool ok = false;
    const cv::Point2f uv_hat = Project(camera, Xc, ok);
    if(!ok)
        return kInf;
    const float dx = uv_hat.x - kp.pt.x;
    const float dy = uv_hat.y - kp.pt.y;
    return std::sqrt(dx * dx + dy * dy);
}

void DIFDynamicStateEstimator::Normalize(Eigen::Vector3f& p)
{
    const float s = p.x() + p.y() + p.z();
    if(!(s > 1e-12f))
    {
        p = Eigen::Vector3f(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);
        return;
    }
    p /= s;
}

float DIFDynamicStateEstimator::UpdateFromFrame(
    const Frame& frame,
    GeometricCamera* camera,
    std::unordered_map<int, DIFTrack>& tracks,
    std::unordered_map<int, std::vector<float>>& out_inst_errors,
    const cv::Mat* label_map,
    const std::vector<int>* local_to_global,
    const std::vector<float>* flow_bg_errors,
    const std::unordered_map<int, std::vector<float>>* flow_inst_errors)
{
    out_inst_errors.clear();
    if(!mCfg.enable)
        return kInf;

    std::vector<float> bg_errors;
    bg_errors.reserve(static_cast<size_t>(frame.N));
    std::vector<float> all_errors;
    all_errors.reserve(static_cast<size_t>(frame.N));

    const bool has_label_map =
        (label_map && !label_map->empty() && label_map->type() == CV_16S && local_to_global && !local_to_global->empty());

    for(int i = 0; i < frame.N; ++i)
    {
        MapPoint* mp = frame.mvpMapPoints[i];
        if(!mp)
            continue;
        if(i < static_cast<int>(frame.mvbOutlier.size()) && frame.mvbOutlier[i])
            continue;

        const cv::KeyPoint& kp = (i < static_cast<int>(frame.mvKeysUn.size())) ? frame.mvKeysUn[i] : frame.mvKeys[i];
        const float r = ReprojectionErrorPx(frame, camera, mp, kp);
        if(!std::isfinite(r))
            continue;

        int tid = mp->mnInstanceId;
        if(tid < 0 && has_label_map)
        {
            const int x = static_cast<int>(kp.pt.x + 0.5f);
            const int y = static_cast<int>(kp.pt.y + 0.5f);
            if(x >= 0 && y >= 0 && x < label_map->cols && y < label_map->rows)
            {
                const int local_id = static_cast<int>(label_map->at<int16_t>(y, x));
                if(local_id >= 0 && local_id < static_cast<int>(local_to_global->size()))
                    tid = (*local_to_global)[static_cast<size_t>(local_id)];
            }
        }

        if(tid >= 0)
            out_inst_errors[tid].push_back(r);
        else
            bg_errors.push_back(r);
        all_errors.push_back(r);
    }

    float r_bg = kInf;
    if(static_cast<int>(bg_errors.size()) >= mCfg.n_bg_min)
        r_bg = Median(bg_errors);
    else if(mCfg.r_bg_fallback_enable && static_cast<int>(all_errors.size()) >= mCfg.n_bg_min)
    {
        std::vector<float> tmp = all_errors;
        r_bg = MedianOfSmallestFraction(tmp, mCfg.r_bg_trim_frac);
    }

    float e_bg = kInf;
    if(mCfg.flow_enable && flow_bg_errors && static_cast<int>(flow_bg_errors->size()) >= mCfg.flow_n_bg_min)
    {
        std::vector<float> tmp = *flow_bg_errors;
        e_bg = MedianOfSmallestFraction(tmp, mCfg.flow_bg_trim_frac);
    }
    else if(mCfg.flow_enable && mCfg.flow_bg_fallback_enable && flow_inst_errors)
    {
        size_t total = 0;
        for(const auto& kv : *flow_inst_errors)
            total += kv.second.size();
        std::vector<float> tmp;
        tmp.reserve(total + (flow_bg_errors ? flow_bg_errors->size() : 0));
        if(flow_bg_errors)
        {
            for(const float e : *flow_bg_errors)
            {
                if(std::isfinite(e))
                    tmp.push_back(e);
            }
        }
        for(const auto& kv : *flow_inst_errors)
        {
            for(const float e : kv.second)
            {
                if(std::isfinite(e))
                    tmp.push_back(e);
            }
        }
        if(static_cast<int>(tmp.size()) >= mCfg.flow_n_bg_min)
            e_bg = MedianOfSmallestFraction(tmp, mCfg.flow_bg_trim_frac);
    }

    // Robust speed "background" level (bias) from the current-update v_obs population.
    // NOTE: With asynchronous segmentation, v_obs may arrive late; we apply each v_obs exactly once when it becomes
    // available, regardless of the current frame_id.
    float v_bg = 0.0f;
    bool has_v_bg = false;
    if(mCfg.v_bg_enable)
    {
        std::vector<float> v_samples;
        v_samples.reserve(tracks.size());
        const double t_now = frame.mTimeStamp;
        for(const auto& kv : tracks)
        {
            const DIFTrack& tr = kv.second;
            if(tr.last_v_obs_frame < 0)
                continue;
            if(tr.last_v_obs_frame == tr.last_v_obs_used_frame)
                continue;
            const double dt = (tr.last_timestamp > 0.0 && t_now > 0.0) ? (t_now - tr.last_timestamp) : 0.0;
            if(mCfg.mask_max_age_s > 0.0f && dt > static_cast<double>(mCfg.mask_max_age_s))
                continue;
            const float v = std::min(std::max(tr.v_obs, 0.0f), mCfg.v_clip);
            if(std::isfinite(v))
                v_samples.push_back(v);
        }
        if(static_cast<int>(v_samples.size()) >= std::max(1, mCfg.v_bg_n_min))
        {
            const float v_est = MedianOfSmallestFraction(v_samples, mCfg.v_bg_trim_frac);
            if(std::isfinite(v_est))
            {
                v_bg = std::max(0.0f, v_est);
                has_v_bg = true;
            }
        }
    }

    const int Kstatic_cfg = std::max(1, mCfg.static_K);

    const bool two_state = (mCfg.state_mode == 1);

    // Pre-normalize 2-state transition matrix (row-wise) for numerical stability.
    float A2[4] = {mCfg.A2[0], mCfg.A2[1], mCfg.A2[2], mCfg.A2[3]};
    float logA2[4] = {0, 0, 0, 0};
    if(two_state)
    {
        for(float& v : A2)
            v = std::max(1e-6f, std::min(1.0f, v));
        const float s0 = A2[0] + A2[1];
        const float s1 = A2[2] + A2[3];
        A2[0] = (s0 > 1e-12f) ? (A2[0] / s0) : 0.5f;
        A2[1] = (s0 > 1e-12f) ? (A2[1] / s0) : 0.5f;
        A2[2] = (s1 > 1e-12f) ? (A2[2] / s1) : 0.5f;
        A2[3] = (s1 > 1e-12f) ? (A2[3] / s1) : 0.5f;
        logA2[0] = SafeLog(A2[0]);
        logA2[1] = SafeLog(A2[1]);
        logA2[2] = SafeLog(A2[2]);
        logA2[3] = SafeLog(A2[3]);
    }

    // Update each track
    for(auto& kv : tracks)
    {
        DIFTrack& tr = kv.second;
        const DIFTrackState prev_state_hat = tr.state_hat;

        if(two_state)
        {
            // -----------------------
            // 2-state HMM {S, D} + map_lock
            // -----------------------
            const Eigen::Vector3f prev3 = tr.alpha;
            float prevS = std::max(0.0f, prev3.x() + prev3.y()); // merge legacy MS probability into S
            float prevD = std::max(0.0f, prev3.z());
            const float psum = prevS + prevD;
            if(psum > 1e-12f)
            {
                prevS /= psum;
                prevD /= psum;
            }
            else
            {
                prevS = 0.5f;
                prevD = 0.5f;
            }

            const float log_prevS = SafeLog(prevS);
            const float log_prevD = SafeLog(prevD);

            // Predict alpha (log-domain): alpha_pred(j) = sum_i alpha_prev(i) A2_ij
            float log_alpha_pred_S = LogSumExp2(log_prevS + logA2[0], log_prevD + logA2[2]); // -> S
            float log_alpha_pred_D = LogSumExp2(log_prevS + logA2[1], log_prevD + logA2[3]); // -> D

            float log_alpha_S = log_alpha_pred_S;
            float log_alpha_D = log_alpha_pred_D;

            // Speed emission: apply each new v_obs exactly once (supports asynchronous/late segmentations).
            const double t_now = frame.mTimeStamp;
            const double dt_obs = (tr.last_timestamp > 0.0 && t_now > 0.0) ? (t_now - tr.last_timestamp) : 0.0;
            const bool has_new_v_obs = (tr.last_v_obs_frame >= 0) && (tr.last_v_obs_frame != tr.last_v_obs_used_frame);
            bool v_obs_applied = false;
            bool v_strong_dynamic = false;
            bool v_static_ok = false;
            if(has_new_v_obs)
            {
                const bool v_obs_stale =
                    (mCfg.mask_max_age_s > 0.0f) && (dt_obs > static_cast<double>(mCfg.mask_max_age_s));
                if(!v_obs_stale)
                {
                    float v = std::min(std::max(tr.v_obs, 0.0f), mCfg.v_clip);
                    if(has_v_bg)
                        v = std::max(0.0f, v - v_bg);
                    const float q_vobs = std::max(0.0f, std::min(1.0f, tr.v_obs_q));
                    v_strong_dynamic = (v >= (mCfg.mu_D + mCfg.sigma_D));
                    const float v_static_thresh = std::max(0.0f, std::min(0.5f * mCfg.mu_D, 2.0f * mCfg.sigma_S));
                    v_static_ok = (v <= v_static_thresh);
                    const float pS = HalfNormalPdf(v, mCfg.sigma_S);
                    const float pD = TruncatedNormalPdfAt0(v, mCfg.mu_D, mCfg.sigma_D);
                    log_alpha_S += q_vobs * SafeLog(pS);
                    log_alpha_D += q_vobs * SafeLog(pD);
                    v_obs_applied = true;
                }
                tr.last_v_obs_used_frame = tr.last_v_obs_frame;
            }

            // Circuit breaker (always update counters; it can also help recover from false D).
            bool breaker_on = false;
            {
                auto it = out_inst_errors.find(tr.id);
                if(it != out_inst_errors.end())
                {
                    std::vector<float>& errs = it->second;
                    if(static_cast<int>(errs.size()) >= mCfg.n_min && std::isfinite(r_bg) && r_bg <= mCfg.tau_bg)
                    {
                        const float r_m = Median(errs);
                        const float q = r_m / (r_bg + mCfg.eps);
                        const float dr = r_m - r_bg;
                        const bool strong =
                            (q >= mCfg.tau_q) &&
                            (dr >= mCfg.tau_dr) &&
                            (!(mCfg.tau_r_m_min > 0.0f) || (r_m >= mCfg.tau_r_m_min));
                        if(strong)
                            tr.err_count++;
                        else
                            tr.err_count = std::max(0, tr.err_count - 1);
                        if(tr.err_count >= mCfg.K)
                            breaker_on = true;

                        if(mCfg.static_enable)
                        {
                            const bool static_ok = (dr <= mCfg.static_tau_dr) || (q <= mCfg.static_tau_q);
                            if(static_ok)
                                tr.static_err_count = std::min(Kstatic_cfg, tr.static_err_count + 1);
                            else
                                tr.static_err_count = std::max(0, tr.static_err_count - 1);
                        }
                    }
                    else
                    {
                        tr.err_count = std::max(0, tr.err_count - 1);
                        tr.static_err_count = std::max(0, tr.static_err_count - 1);
                    }
                }
                else
                {
                    tr.err_count = std::max(0, tr.err_count - 1);
                    tr.static_err_count = std::max(0, tr.static_err_count - 1);
                }
            }

            bool flow_breaker_on = false;
            if(mCfg.flow_enable && flow_inst_errors)
            {
                auto it2 = flow_inst_errors->find(tr.id);
                if(it2 != flow_inst_errors->end())
                {
                    const std::vector<float>& errs = it2->second;
                    constexpr int kFlowMinSamplesAdaptive = 5;
                    const int n_flow = static_cast<int>(errs.size());
                    const bool enough_flow = (n_flow >= mCfg.flow_n_min);
                    const bool enough_flow_adapt = (!enough_flow && n_flow >= kFlowMinSamplesAdaptive);
                    if((enough_flow || enough_flow_adapt) && std::isfinite(e_bg) && e_bg <= mCfg.flow_tau_bg)
                    {
                        std::vector<float> tmp = errs;
                        const float e_m = Median(tmp);
                        const float q = e_m / (e_bg + mCfg.eps);
                        const float de = e_m - e_bg;
                        float q_thr = mCfg.flow_tau_q;
                        float de_thr = mCfg.flow_tau_de;
                        if(enough_flow_adapt && mCfg.flow_n_min > 0)
                        {
                            const float scale = std::sqrt(static_cast<float>(mCfg.flow_n_min) / static_cast<float>(std::max(1, n_flow)));
                            q_thr *= std::max(1.0f, scale);
                            de_thr *= std::max(1.0f, scale);
                        }
                        const bool strong =
                            (q >= q_thr) &&
                            (de >= de_thr) &&
                            (!(mCfg.flow_tau_e_m_min > 0.0f) || (e_m >= mCfg.flow_tau_e_m_min));
                        if(strong)
                            tr.flow_err_count++;
                        else
                            tr.flow_err_count = std::max(0, tr.flow_err_count - 1);
                        if(tr.flow_err_count >= mCfg.flow_K)
                            flow_breaker_on = true;

                        if(mCfg.static_enable && enough_flow)
                        {
                            const bool static_ok = (de <= mCfg.static_flow_tau_de) || (q <= mCfg.static_flow_tau_q);
                            if(static_ok)
                                tr.static_flow_count = std::min(Kstatic_cfg, tr.static_flow_count + 1);
                            else
                                tr.static_flow_count = std::max(0, tr.static_flow_count - 1);
                        }
                    }
                    else
                    {
                        tr.flow_err_count = std::max(0, tr.flow_err_count - 1);
                        tr.static_flow_count = std::max(0, tr.static_flow_count - 1);
                    }
                }
                else
                {
                    tr.flow_err_count = std::max(0, tr.flow_err_count - 1);
                    tr.static_flow_count = std::max(0, tr.static_flow_count - 1);
                }
            }
            else
            {
                tr.flow_err_count = std::max(0, tr.flow_err_count - 1);
                tr.static_flow_count = std::max(0, tr.static_flow_count - 1);
            }

            // Static evidence (speed)
            if(mCfg.static_enable)
            {
                if(v_obs_applied)
                {
                    if(v_static_ok)
                        tr.static_v_count = std::min(Kstatic_cfg, tr.static_v_count + 1);
                    else
                        tr.static_v_count = std::max(0, tr.static_v_count - 1);
                }
                else
                {
                    tr.static_v_count = std::max(0, tr.static_v_count - 1);
                }
            }
            else
            {
                tr.static_v_count = 0;
            }

            const bool force_dynamic = breaker_on || flow_breaker_on;
            if(force_dynamic)
            {
                log_alpha_S += SafeLog(mCfg.lambda_other);
                log_alpha_D += SafeLog(mCfg.lambda_D);
            }

            bool static_lock_on =
                mCfg.static_enable &&
                (tr.static_err_count >= std::max(1, mCfg.static_K) ||
                 tr.static_flow_count >= std::max(1, mCfg.static_K) ||
                 tr.static_v_count >= std::max(1, mCfg.static_K));
            if(v_obs_applied && !v_static_ok)
                static_lock_on = false;
            if(mCfg.static_enable)
            {
                const int Kstatic = std::max(1, mCfg.static_K);
                float wr = std::min(1.0f, static_cast<float>(std::max(0, tr.static_err_count)) / static_cast<float>(Kstatic));
                float wf = std::min(1.0f, static_cast<float>(std::max(0, tr.static_flow_count)) / static_cast<float>(Kstatic));
                const float wv = std::min(1.0f, static_cast<float>(std::max(0, tr.static_v_count)) / static_cast<float>(Kstatic));
                if(v_obs_applied && !v_static_ok)
                {
                    wr = 0.0f;
                    wf = 0.0f;
                }
                const float w = std::max(wv, std::max(wr, wf));
                if(w > 0.0f)
                {
                    log_alpha_S += w * SafeLog(mCfg.static_lambda_S);
                    log_alpha_D += w * SafeLog(mCfg.static_lambda_D);
                }
            }

            // Normalize log_alpha -> alpha
            const float lse = LogSumExp2(log_alpha_S, log_alpha_D);
            log_alpha_S -= lse;
            log_alpha_D -= lse;
            float alphaS = std::exp(log_alpha_S);
            float alphaD = std::exp(log_alpha_D);
            const float asum = alphaS + alphaD;
            if(asum > 1e-12f)
            {
                alphaS /= asum;
                alphaD /= asum;
            }
            else
            {
                alphaS = 0.5f;
                alphaD = 0.5f;
            }
            tr.alpha = Eigen::Vector3f(alphaS, 0.0f, alphaD);

            // Hysteresis output + maturity gate
            float tau_up = mCfg.tau_up;
            float tau_down = mCfg.tau_down;

            if(mCfg.static_enable && !force_dynamic)
            {
                const int Kstatic = std::max(1, mCfg.static_K);
                float progress_r = std::min(1.0f, static_cast<float>(std::max(0, tr.static_err_count)) / static_cast<float>(Kstatic));
                float progress_f = std::min(1.0f, static_cast<float>(std::max(0, tr.static_flow_count)) / static_cast<float>(Kstatic));
                const float progress_v = std::min(1.0f, static_cast<float>(std::max(0, tr.static_v_count)) / static_cast<float>(Kstatic));
                if(v_obs_applied && !v_static_ok)
                {
                    progress_r = 0.0f;
                    progress_f = 0.0f;
                }
                const float progress = std::max(progress_v, std::max(progress_r, progress_f));
                if(progress > 0.0f)
                {
                    tau_up = mCfg.tau_up + progress * (mCfg.static_hyst_tau_up - mCfg.tau_up);
                    tau_down = mCfg.tau_down + progress * (mCfg.static_hyst_tau_down - mCfg.tau_down);
                }
            }

            if(static_lock_on && !force_dynamic)
            {
                tau_up = std::max(tau_up, mCfg.static_hyst_tau_up);
                tau_down = std::max(tau_down, mCfg.static_hyst_tau_down);
            }
            tau_up = std::max(0.0f, std::min(1.0f, tau_up));
            tau_down = std::max(0.0f, std::min(1.0f, tau_down));

            if(static_lock_on && !force_dynamic)
            {
                tr.enter_dyn_count = 0;
                tr.state_hat = DIFTrackState::S;
            }
            else
            {
                const float aD = tr.alpha.z();
                const int enterK = std::max(1, mCfg.enter_D_K);
                if(tr.state_hat != DIFTrackState::D && aD >= tau_up)
                {
                    const bool mature_ok = (mCfg.mature_min <= 0) || (tr.mature_count >= mCfg.mature_min);
                    if(mature_ok || force_dynamic || (v_obs_applied && v_strong_dynamic))
                    {
                        if(force_dynamic || (v_obs_applied && v_strong_dynamic) || enterK <= 1)
                        {
                            tr.enter_dyn_count = 0;
                            tr.state_hat = DIFTrackState::D;
                            tr.ever_dynamic = true;
                        }
                        else
                        {
                            if(v_obs_applied)
                                tr.enter_dyn_count++;
                            if(tr.enter_dyn_count >= enterK)
                            {
                                tr.enter_dyn_count = 0;
                                tr.state_hat = DIFTrackState::D;
                                tr.ever_dynamic = true;
                            }
                            else
                            {
                                tr.state_hat = DIFTrackState::S;
                            }
                        }
                    }
                    else
                    {
                        tr.enter_dyn_count = 0;
                        tr.state_hat = DIFTrackState::S;
                    }
                }
                else if(tr.state_hat == DIFTrackState::D && aD <= tau_down)
                {
                    tr.enter_dyn_count = 0;
                    tr.state_hat = DIFTrackState::S;
                }
                else if(tr.state_hat != DIFTrackState::D)
                {
                    tr.enter_dyn_count = 0;
                    tr.state_hat = DIFTrackState::S;
                }
                else
                {
                    tr.enter_dyn_count = 0;
                }
            }

            // Map gating update (state_mode=1).
            if(!mCfg.map_lock_enable)
            {
                tr.map_lock = false;
                tr.map_S_streak = 0;
                tr.map_cooldown = 0;
                tr.w_lock = 1.0f;
            }
            else
            {
                const int N_confirm = std::max(1, mCfg.map_confirm_N);
                const int N_cooldown = std::max(0, mCfg.map_cooldown_N);

                if(prev_state_hat == DIFTrackState::D && tr.state_hat == DIFTrackState::S)
                {
                    tr.map_cooldown = N_cooldown;
                    tr.map_S_streak = 0;
                }

                if(tr.state_hat == DIFTrackState::D)
                {
                    tr.map_S_streak = 0;
                }
                else
                {
                    if(tr.map_cooldown > 0)
                    {
                        tr.map_cooldown = std::max(0, tr.map_cooldown - 1);
                        tr.map_S_streak = 0;
                    }
                    else
                    {
                        tr.map_S_streak += 1;
                    }
                }

                tr.map_lock = (tr.map_S_streak < N_confirm) || (tr.map_cooldown > 0);

                // DIF-SLAM v2.0 lock weight schedule:
                // w_lock = clip(map_S_streak / N_confirm, w_lock_min, 1).
                const float w_min = std::max(0.0f, std::min(1.0f, mCfg.w_lock_min));
                float w = 1.0f;
                if(tr.map_lock)
                {
                    const float ratio = static_cast<float>(std::max(0, tr.map_S_streak)) / static_cast<float>(N_confirm);
                    w = std::max(w_min, std::min(1.0f, ratio));
                }
                tr.w_lock = w;
            }

            continue;
        }

        // Predict alpha (log-domain): alpha_pred(j) = sum_i alpha_prev(i) A_ij
        const Eigen::Vector3f prev = tr.alpha;
        const Eigen::Vector3f log_prev(SafeLog(prev.x()), SafeLog(prev.y()), SafeLog(prev.z()));
        const float logA[9] = {
            SafeLog(mCfg.A[0]), SafeLog(mCfg.A[1]), SafeLog(mCfg.A[2]),
            SafeLog(mCfg.A[3]), SafeLog(mCfg.A[4]), SafeLog(mCfg.A[5]),
            SafeLog(mCfg.A[6]), SafeLog(mCfg.A[7]), SafeLog(mCfg.A[8])
        };

        Eigen::Vector3f log_alpha_pred;
        // j=0 (S)
        log_alpha_pred.x() = LogSumExp3(Eigen::Vector3f(log_prev.x() + logA[0], log_prev.y() + logA[3], log_prev.z() + logA[6]));
        // j=1 (MS)
        log_alpha_pred.y() = LogSumExp3(Eigen::Vector3f(log_prev.x() + logA[1], log_prev.y() + logA[4], log_prev.z() + logA[7]));
        // j=2 (D)
        log_alpha_pred.z() = LogSumExp3(Eigen::Vector3f(log_prev.x() + logA[2], log_prev.y() + logA[5], log_prev.z() + logA[8]));

        Eigen::Vector3f log_alpha = log_alpha_pred;

        // Speed emission: apply each new v_obs exactly once (supports asynchronous/late segmentations).
        const double t_now = frame.mTimeStamp;
        const double dt_obs = (tr.last_timestamp > 0.0 && t_now > 0.0) ? (t_now - tr.last_timestamp) : 0.0;
        const bool has_new_v_obs = (tr.last_v_obs_frame >= 0) && (tr.last_v_obs_frame != tr.last_v_obs_used_frame);
        bool v_obs_applied = false;
        bool v_strong_dynamic = false;
        bool v_static_ok = false;
        if(has_new_v_obs)
        {
            // Drop very stale v_obs (e.g., late arrival after long stalls / warm-up) to avoid corrupting states.
            const bool v_obs_stale =
                (mCfg.mask_max_age_s > 0.0f) && (dt_obs > static_cast<double>(mCfg.mask_max_age_s));
            if(!v_obs_stale)
            {
                float v = std::min(std::max(tr.v_obs, 0.0f), mCfg.v_clip);
                if(has_v_bg)
                    v = std::max(0.0f, v - v_bg);
                const float q_vobs = std::max(0.0f, std::min(1.0f, tr.v_obs_q));
                // Strong-motion fast path: if v is clearly in the dynamic regime, allow faster entry into D.
                v_strong_dynamic = (v >= (mCfg.mu_D + mCfg.sigma_D));
                const float v_static_thresh = std::max(0.0f, std::min(0.5f * mCfg.mu_D, 2.0f * mCfg.sigma_MS));
                v_static_ok = (v <= v_static_thresh);
                const float pS = HalfNormalPdf(v, mCfg.sigma_S);
                const float pMS = HalfNormalPdf(v, mCfg.sigma_MS);
                const float pD = TruncatedNormalPdfAt0(v, mCfg.mu_D, mCfg.sigma_D);
                // Soft update (quality-weighted): q_vobs in [0,1] scales the emission impact.
                log_alpha.x() += q_vobs * SafeLog(pS);
                log_alpha.y() += q_vobs * SafeLog(pMS);
                log_alpha.z() += q_vobs * SafeLog(pD);
                v_obs_applied = true;
            }
            tr.last_v_obs_used_frame = tr.last_v_obs_frame;
        }
        // Circuit breaker (always update counters; it can also help recover from false D).
        bool breaker_on = false;
        {
            auto it = out_inst_errors.find(tr.id);
            if(it != out_inst_errors.end())
            {
                std::vector<float>& errs = it->second;
                if(static_cast<int>(errs.size()) >= mCfg.n_min && std::isfinite(r_bg) && r_bg <= mCfg.tau_bg)
                {
                    const float r_m = Median(errs);
                    const float q = r_m / (r_bg + mCfg.eps);
                    const float dr = r_m - r_bg;
                    const bool strong =
                        (q >= mCfg.tau_q) &&
                        (dr >= mCfg.tau_dr) &&
                        (!(mCfg.tau_r_m_min > 0.0f) || (r_m >= mCfg.tau_r_m_min));
                    if(strong)
                        tr.err_count++;
                    else
                        tr.err_count = std::max(0, tr.err_count - 1);
                    if(tr.err_count >= mCfg.K)
                        breaker_on = true;

                    // Static evidence (reprojection): instance behaves like background.
                    if(mCfg.static_enable)
                    {
                        const bool static_ok = (dr <= mCfg.static_tau_dr) || (q <= mCfg.static_tau_q);
                        if(static_ok)
                            tr.static_err_count = std::min(Kstatic_cfg, tr.static_err_count + 1);
                        else
                            tr.static_err_count = std::max(0, tr.static_err_count - 1);
                    }
                }
                else
                {
                    tr.err_count = std::max(0, tr.err_count - 1);
                    tr.static_err_count = std::max(0, tr.static_err_count - 1);
                }
            }
            else
            {
                tr.err_count = std::max(0, tr.err_count - 1);
                tr.static_err_count = std::max(0, tr.static_err_count - 1);
            }
        }

        bool flow_breaker_on = false;
        if(mCfg.flow_enable && flow_inst_errors)
        {
            auto it2 = flow_inst_errors->find(tr.id);
            if(it2 != flow_inst_errors->end())
            {
                const std::vector<float>& errs = it2->second;
                // Adaptive small-sample support: for low-texture moving objects, the number of reliable descriptor
                // matches can be < flow_n_min. We still want to trigger the breaker when the evidence is very strong,
                // but require stricter thresholds when sample count is small to avoid false positives.
                constexpr int kFlowMinSamplesAdaptive = 5;
                const int n_flow = static_cast<int>(errs.size());
                const bool enough_flow = (n_flow >= mCfg.flow_n_min);
                const bool enough_flow_adapt = (!enough_flow && n_flow >= kFlowMinSamplesAdaptive);
                if((enough_flow || enough_flow_adapt) && std::isfinite(e_bg) && e_bg <= mCfg.flow_tau_bg)
                {
                    std::vector<float> tmp = errs;
                    const float e_m = Median(tmp);
                    const float q = e_m / (e_bg + mCfg.eps);
                    const float de = e_m - e_bg;
                    float q_thr = mCfg.flow_tau_q;
                    float de_thr = mCfg.flow_tau_de;
                    if(enough_flow_adapt && mCfg.flow_n_min > 0)
                    {
                        const float scale = std::sqrt(static_cast<float>(mCfg.flow_n_min) / static_cast<float>(std::max(1, n_flow)));
                        q_thr *= std::max(1.0f, scale);
                        de_thr *= std::max(1.0f, scale);
                    }
                    const bool strong =
                        (q >= q_thr) &&
                        (de >= de_thr) &&
                        (!(mCfg.flow_tau_e_m_min > 0.0f) || (e_m >= mCfg.flow_tau_e_m_min));
                    if(strong)
                        tr.flow_err_count++;
                    else
                        tr.flow_err_count = std::max(0, tr.flow_err_count - 1);
                    if(tr.flow_err_count >= mCfg.flow_K)
                        flow_breaker_on = true;

                    // Static evidence (flow): instance behaves like background.
                    // Only accumulate static evidence when sample size is sufficient (avoid locking on few matches).
                    if(mCfg.static_enable && enough_flow)
                    {
                        const bool static_ok = (de <= mCfg.static_flow_tau_de) || (q <= mCfg.static_flow_tau_q);
                        if(static_ok)
                            tr.static_flow_count = std::min(Kstatic_cfg, tr.static_flow_count + 1);
                        else
                            tr.static_flow_count = std::max(0, tr.static_flow_count - 1);
                    }
                }
                else
                {
                    tr.flow_err_count = std::max(0, tr.flow_err_count - 1);
                    tr.static_flow_count = std::max(0, tr.static_flow_count - 1);
                }
            }
            else
            {
                tr.flow_err_count = std::max(0, tr.flow_err_count - 1);
                tr.static_flow_count = std::max(0, tr.static_flow_count - 1);
            }
        }

        // Static evidence (speed): if v_obs is consistently tiny after bias removal, treat the instance as static even
        // when per-instance reprojection/flow residuals are unavailable (common for low-texture background segments).
        if(mCfg.static_enable)
        {
            if(v_obs_applied)
            {
                if(v_static_ok)
                    tr.static_v_count = std::min(Kstatic_cfg, tr.static_v_count + 1);
                else
                    tr.static_v_count = std::max(0, tr.static_v_count - 1);
            }
            else
            {
                tr.static_v_count = std::max(0, tr.static_v_count - 1);
            }
        }
        else
        {
            tr.static_v_count = 0;
        }

        const bool force_dynamic = breaker_on || flow_breaker_on;
        if(force_dynamic)
        {
            log_alpha.x() += SafeLog(mCfg.lambda_other);
            log_alpha.y() += SafeLog(mCfg.lambda_other);
            log_alpha.z() += SafeLog(mCfg.lambda_D);
        }

        // Static evidence suppression: down-weight D when instance residual matches background consistently.
        // IMPORTANT: static evidence must NOT "freeze" an actually moving object in S/MS.
        // If the current-frame speed observation indicates motion, we bypass the full static lock
        // and also prevent residual-based static evidence from dominating the suppression weight.
        bool static_lock_on =
            mCfg.static_enable &&
            (tr.static_err_count >= std::max(1, mCfg.static_K) ||
             tr.static_flow_count >= std::max(1, mCfg.static_K) ||
             tr.static_v_count >= std::max(1, mCfg.static_K));
        if(v_obs_applied && !v_static_ok)
            static_lock_on = false;
        if(mCfg.static_enable)
        {
            const int Kstatic = std::max(1, mCfg.static_K);
            float wr = std::min(1.0f, static_cast<float>(std::max(0, tr.static_err_count)) / static_cast<float>(Kstatic));
            float wf = std::min(1.0f, static_cast<float>(std::max(0, tr.static_flow_count)) / static_cast<float>(Kstatic));
            const float wv = std::min(1.0f, static_cast<float>(std::max(0, tr.static_v_count)) / static_cast<float>(Kstatic));
            // When motion is observed (speed not static), do not let residual-based "static" evidence
            // keep suppressing D. This is the main fix for "starts moving but still S".
            if(v_obs_applied && !v_static_ok)
            {
                wr = 0.0f;
                wf = 0.0f;
            }
            const float w = std::max(wv, std::max(wr, wf));
            if(w > 0.0f)
            {
                log_alpha.x() += w * SafeLog(mCfg.static_lambda_S);
                log_alpha.y() += w * SafeLog(mCfg.static_lambda_MS);
                log_alpha.z() += w * SafeLog(mCfg.static_lambda_D);
            }
        }

        // Normalize log_alpha -> alpha
        const float lse = LogSumExp3(log_alpha);
        log_alpha.array() -= lse;
        Eigen::Vector3f alpha(std::exp(log_alpha.x()), std::exp(log_alpha.y()), std::exp(log_alpha.z()));
        Normalize(alpha);
        tr.alpha = alpha;

        // Hysteresis output + maturity gate (Module C 3.5.6)
        // 2026-01-06: Progressive static protection - gradually increase tau_up based on accumulated
        // static evidence even before reaching static_K, to provide early protection for static instances.
        float tau_up = mCfg.tau_up;
        float tau_down = mCfg.tau_down;

        // Progressive static protection: interpolate tau_up based on static evidence progress
        if(mCfg.static_enable && !force_dynamic)
        {
            const int Kstatic = std::max(1, mCfg.static_K);
            float progress_r = std::min(1.0f, static_cast<float>(std::max(0, tr.static_err_count)) / static_cast<float>(Kstatic));
            float progress_f = std::min(1.0f, static_cast<float>(std::max(0, tr.static_flow_count)) / static_cast<float>(Kstatic));
            const float progress_v = std::min(1.0f, static_cast<float>(std::max(0, tr.static_v_count)) / static_cast<float>(Kstatic));
            if(v_obs_applied && !v_static_ok)
            {
                // Same rationale as above: once motion is observed, do not boost tau_up/tau_down by
                // residual-based static evidence.
                progress_r = 0.0f;
                progress_f = 0.0f;
            }
            const float progress = std::max(progress_v, std::max(progress_r, progress_f));

            // Interpolate tau_up from base value toward static_hyst_tau_up based on progress
            if(progress > 0.0f)
            {
                tau_up = mCfg.tau_up + progress * (mCfg.static_hyst_tau_up - mCfg.tau_up);
                tau_down = mCfg.tau_down + progress * (mCfg.static_hyst_tau_down - mCfg.tau_down);
            }
        }

        // Full static lock: use maximum thresholds
        if(static_lock_on && !force_dynamic)
        {
            tau_up = std::max(tau_up, mCfg.static_hyst_tau_up);
            tau_down = std::max(tau_down, mCfg.static_hyst_tau_down);
        }
        tau_up = std::max(0.0f, std::min(1.0f, tau_up));
        tau_down = std::max(0.0f, std::min(1.0f, tau_down));

        // If the instance has strong static evidence, never enter D unless a breaker triggers.
        // This is a key anti-flicker rule to prevent static background objects from oscillating S/MS/D due to
        // segmentation/centroid jitter in speed observations.
        if(static_lock_on && !force_dynamic)
        {
            tr.enter_dyn_count = 0;
            // If a track was previously marked ever_dynamic, keep it at MS; otherwise lock to S to match "static
            // background should be S" expectation.
            tr.state_hat = tr.ever_dynamic ? DIFTrackState::MS : DIFTrackState::S;
            tr.map_lock = (tr.state_hat == DIFTrackState::MS);
            continue;
        }

        const int enterK = std::max(1, mCfg.enter_D_K);
        if(tr.state_hat != DIFTrackState::D && alpha.z() >= tau_up)
        {
            const bool mature_ok = (mCfg.mature_min <= 0) || (tr.mature_count >= mCfg.mature_min);
            if(mature_ok || force_dynamic || (v_obs_applied && v_strong_dynamic))
            {
                if(force_dynamic || (v_obs_applied && v_strong_dynamic) || enterK <= 1)
                {
                    tr.enter_dyn_count = 0;
                    tr.state_hat = DIFTrackState::D;
                    tr.ever_dynamic = true;
                }
                else
                {
                    // Only count confirmations when a new speed observation is actually applied; UpdateFromFrame runs
                    // every frame, but v_obs typically updates only on segmentation frames.
                    if(v_obs_applied)
                        tr.enter_dyn_count++;
                    if(tr.enter_dyn_count >= enterK)
                    {
                        tr.enter_dyn_count = 0;
                        tr.state_hat = DIFTrackState::D;
                        tr.ever_dynamic = true;
                    }
                    else
                    {
                        tr.state_hat = DIFTrackState::MS;
                    }
                }
            }
            else
            {
                tr.enter_dyn_count = 0;
                tr.state_hat = DIFTrackState::MS;
            }
        }
        else if(tr.state_hat == DIFTrackState::D && alpha.z() <= tau_down)
        {
            tr.enter_dyn_count = 0;
            tr.state_hat = tr.ever_dynamic ? DIFTrackState::MS : ((alpha.x() >= alpha.y()) ? DIFTrackState::S : DIFTrackState::MS);
        }
        else if(tr.state_hat != DIFTrackState::D)
        {
            tr.enter_dyn_count = 0;
            tr.state_hat = (alpha.x() >= alpha.y()) ? DIFTrackState::S : DIFTrackState::MS;
            if(tr.ever_dynamic)
                tr.state_hat = DIFTrackState::MS;
        }
        else
        {
            // Stay in D.
            tr.enter_dyn_count = 0;
        }

        // Legacy mode: map_lock matches MS semantics (forbid map writing, allow optimization).
        tr.map_lock = (tr.state_hat == DIFTrackState::MS);
    }

    return r_bg;
}

cv::Mat DIFDynamicStateEstimator::BuildDynamicMask(int height, int width, const std::unordered_map<int, DIFTrack>& tracks) const
{
    cv::Mat dyn(height, width, CV_8U, cv::Scalar(0));
    if(height <= 0 || width <= 0)
        return dyn;

    for(const auto& kv : tracks)
    {
        const DIFTrack& tr = kv.second;
        if(tr.state_hat != DIFTrackState::D && !(mCfg.dyn_mask_include_ever_dynamic && tr.ever_dynamic))
            continue;
        if(tr.mask_last.Empty())
            continue;

        const DIFBbox& b = tr.mask_last.bbox;
        if(!b.IsValid())
            continue;
        if(b.x1 < 0 || b.y1 < 0 || b.x2 >= width || b.y2 >= height)
            continue;

        cv::Mat roi = dyn(cv::Rect(b.x1, b.y1, b.Width(), b.Height()));
        cv::Mat m = tr.mask_last.mask;
        if(m.type() != CV_8U)
            m.convertTo(m, CV_8U);
        // mask stored as {0,1}; convert to {0,255} on the fly via compare.
        cv::Mat nz = (m != 0);
        roi.setTo(255, nz);
    }

    if(mCfg.mask_dilate > 0)
    {
        const int k = std::max(1, mCfg.mask_dilate);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        cv::dilate(dyn, dyn, kernel);
    }
    return dyn;
}

} // namespace ORB_SLAM3
