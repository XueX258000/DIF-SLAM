/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdint>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include<mutex>

namespace ORB_SLAM3
{

namespace {

cv::Scalar DIFBgrFromHsv(float hue_deg, float sat, float val)
{
    // hue_deg: [0,360), sat/val: [0,1]
    const float h = std::fmod(std::fmod(hue_deg, 360.f) + 360.f, 360.f);
    const float s = std::max(0.f, std::min(1.f, sat));
    const float v = std::max(0.f, std::min(1.f, val));

    const float c = v * s;
    const float hh = h / 60.f;
    const float x = c * (1.f - std::fabs(std::fmod(hh, 2.f) - 1.f));
    const float m = v - c;

    float rp = 0.f, gp = 0.f, bp = 0.f;
    if(0.f <= hh && hh < 1.f)
    {
        rp = c; gp = x; bp = 0.f;
    }
    else if(1.f <= hh && hh < 2.f)
    {
        rp = x; gp = c; bp = 0.f;
    }
    else if(2.f <= hh && hh < 3.f)
    {
        rp = 0.f; gp = c; bp = x;
    }
    else if(3.f <= hh && hh < 4.f)
    {
        rp = 0.f; gp = x; bp = c;
    }
    else if(4.f <= hh && hh < 5.f)
    {
        rp = x; gp = 0.f; bp = c;
    }
    else
    {
        rp = c; gp = 0.f; bp = x;
    }

    const float r = (rp + m) * 255.f;
    const float g = (gp + m) * 255.f;
    const float b = (bp + m) * 255.f;
    return cv::Scalar(std::max(0.f, std::min(255.f, b)),
                      std::max(0.f, std::min(255.f, g)),
                      std::max(0.f, std::min(255.f, r)));
}

cv::Scalar DIFColorFromId(int id)
{
    // Deterministic track_id -> hue mapping (stable across frames).
    const uint32_t x = static_cast<uint32_t>(id) * 2654435761u;
    const float hue = static_cast<float>(x % 360u);
    return DIFBgrFromHsv(hue, 0.75f, 0.95f);
}

cv::Scalar DIFDarken(const cv::Scalar& bgr, float factor)
{
    const float f = std::max(0.f, std::min(1.f, factor));
    return cv::Scalar(std::max(0.0, std::min(255.0, bgr[0] * f)),
                      std::max(0.0, std::min(255.0, bgr[1] * f)),
                      std::max(0.0, std::min(255.0, bgr[2] * f)));
}

const char* DIFStateShort(DIFTrackState st)
{
    switch(st)
    {
    case DIFTrackState::S:
        return "S";
    case DIFTrackState::MS:
        return "MS";
    case DIFTrackState::D:
    default:
        return "D";
    }
}

struct DIFInstanceStat
{
    int area = 0;
    int minx = std::numeric_limits<int>::max();
    int miny = std::numeric_limits<int>::max();
    int maxx = -1;
    int maxy = -1;
    int64_t sumx = 0;
    int64_t sumy = 0;

    cv::Rect Rect() const
    {
        if(area <= 0 || maxx < minx || maxy < miny)
            return cv::Rect();
        return cv::Rect(minx, miny, maxx - minx + 1, maxy - miny + 1);
    }
};

} // namespace

FrameDrawer::FrameDrawer(Atlas* pAtlas):both(false),mpAtlas(pAtlas)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    mImRight = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    mLastFrameId = -1;
}

cv::Mat FrameDrawer::DrawFrame(float imageScale)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    vector<pair<cv::Point2f, cv::Point2f> > vTracks;
    int state; // Tracking state

    Frame currentFrame;
    vector<MapPoint*> vpLocalMap;
    vector<cv::KeyPoint> vMatchesKeys;
    vector<MapPoint*> vpMatchedMPs;
    vector<cv::KeyPoint> vOutlierKeys;
    vector<MapPoint*> vpOutlierMPs;
    map<long unsigned int, cv::Point2f> mProjectPoints;
    map<long unsigned int, cv::Point2f> mMatchedInImage;

    cv::Scalar standardColor(0,255,0);
    cv::Scalar odometryColor(255,0,0);

    bool showDIFOverlay = false;
    int difOverlayMode = 0;
    int difSegFrameId = -1;
    cv::Mat difLabel;
    int difLocalToGlobalFrameId = -1;
    std::vector<int> difLocalToGlobal;
    std::unordered_map<int, DIFTrackState> difTrackStates;
    cv::Mat difDynMask;
    int difFilterMaxLag = 2;
    bool difViewerShowKeypoints = true;
    int difViewerStateStyle = 1;
    bool difViewerInstFillEnable = true;
    bool difViewerInstContourEnable = true;
    bool difViewerInstContourLockEnable = true;
    bool difViewerInstLabelEnable = true;
    bool difViewerHudEnable = true;
    float difViewerAlphaS = 0.16f;
    float difViewerAlphaLock = 0.22f;
    float difViewerAlphaD = 0.35f;
    float difViewerLabelMinAreaRatio = 0.0f;
    float difViewerLabelMinAreaRatioS = 0.0f;

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
            vTracks = mvTracks;

            // DIF: Show segmentation overlay during initialization if available
            showDIFOverlay = mbShowDIFOverlay;
            difOverlayMode = mDIFOverlayMode;
            difSegFrameId = mDIFSegFrameId;
            difLabel = mDIFLabelMap;
            difLocalToGlobalFrameId = mDIFLocalToGlobalFrameId;
            difLocalToGlobal = mDIFLocalToGlobal;
            difTrackStates = mDIFTrackStates;
            difDynMask = mDIFDynMask;
            difFilterMaxLag = mDIFFilterMaxLag;
            difViewerShowKeypoints = mDIFViewerShowKeypoints;
            difViewerStateStyle = mDIFViewerStateStyle;
            difViewerInstFillEnable = mDIFViewerInstFillEnable;
            difViewerInstContourEnable = mDIFViewerInstContourEnable;
            difViewerInstContourLockEnable = mDIFViewerInstContourLockEnable;
            difViewerInstLabelEnable = mDIFViewerInstLabelEnable;
            difViewerHudEnable = mDIFViewerHudEnable;
            difViewerAlphaS = mDIFViewerAlphaS;
            difViewerAlphaLock = mDIFViewerAlphaLock;
            difViewerAlphaD = mDIFViewerAlphaD;
            difViewerLabelMinAreaRatio = mDIFViewerLabelMinAreaRatio;
            difViewerLabelMinAreaRatioS = mDIFViewerLabelMinAreaRatioS;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;

            currentFrame = mCurrentFrame;
            vpLocalMap = mvpLocalMap;
            vMatchesKeys = mvMatchedKeys;
            vpMatchedMPs = mvpMatchedMPs;
            vOutlierKeys = mvOutlierKeys;
            vpOutlierMPs = mvpOutlierMPs;
            mProjectPoints = mmProjectPoints;
            mMatchedInImage = mmMatchedInImage;

            showDIFOverlay = mbShowDIFOverlay;
            difOverlayMode = mDIFOverlayMode;
            difSegFrameId = mDIFSegFrameId;
            difLabel = mDIFLabelMap;
            difLocalToGlobalFrameId = mDIFLocalToGlobalFrameId;
            difLocalToGlobal = mDIFLocalToGlobal;
            difTrackStates = mDIFTrackStates;
            difDynMask = mDIFDynMask;
            difFilterMaxLag = mDIFFilterMaxLag;
            difViewerShowKeypoints = mDIFViewerShowKeypoints;
            difViewerStateStyle = mDIFViewerStateStyle;
            difViewerInstFillEnable = mDIFViewerInstFillEnable;
            difViewerInstContourEnable = mDIFViewerInstContourEnable;
            difViewerInstContourLockEnable = mDIFViewerInstContourLockEnable;
            difViewerInstLabelEnable = mDIFViewerInstLabelEnable;
            difViewerHudEnable = mDIFViewerHudEnable;
            difViewerAlphaS = mDIFViewerAlphaS;
            difViewerAlphaLock = mDIFViewerAlphaLock;
            difViewerAlphaD = mDIFViewerAlphaD;
            difViewerLabelMinAreaRatio = mDIFViewerLabelMinAreaRatio;
            difViewerLabelMinAreaRatioS = mDIFViewerLabelMinAreaRatioS;

        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;

            // DIF: Also show segmentation overlay when tracking is lost.
            // Segmentation runs independently and should be visualized regardless of tracking state.
            showDIFOverlay = mbShowDIFOverlay;
            difOverlayMode = mDIFOverlayMode;
            difSegFrameId = mDIFSegFrameId;
            difLabel = mDIFLabelMap;
            difLocalToGlobalFrameId = mDIFLocalToGlobalFrameId;
            difLocalToGlobal = mDIFLocalToGlobal;
            difTrackStates = mDIFTrackStates;
            difDynMask = mDIFDynMask;
            difFilterMaxLag = mDIFFilterMaxLag;
            difViewerShowKeypoints = mDIFViewerShowKeypoints;
            difViewerStateStyle = mDIFViewerStateStyle;
            difViewerInstFillEnable = mDIFViewerInstFillEnable;
            difViewerInstContourEnable = mDIFViewerInstContourEnable;
            difViewerInstContourLockEnable = mDIFViewerInstContourLockEnable;
            difViewerInstLabelEnable = mDIFViewerInstLabelEnable;
            difViewerHudEnable = mDIFViewerHudEnable;
            difViewerAlphaS = mDIFViewerAlphaS;
            difViewerAlphaLock = mDIFViewerAlphaLock;
            difViewerAlphaD = mDIFViewerAlphaD;
            difViewerLabelMinAreaRatio = mDIFViewerLabelMinAreaRatio;
            difViewerLabelMinAreaRatioS = mDIFViewerLabelMinAreaRatioS;
        }
    }

    if(imageScale != 1.f)
    {
        int imWidth = im.cols / imageScale;
        int imHeight = im.rows / imageScale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    if(showDIFOverlay && difOverlayMode == 1)
    {
        if(difDynMask.empty())
        {
            if(difViewerHudEnable)
            {
                cv::putText(im, "NO_MASK", cv::Point(10, 28), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(0, 0, 0), 3, 8);
                cv::putText(im, "NO_MASK", cv::Point(10, 28), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(255, 255, 255), 1, 8);
            }
        }
        else
        {
            cv::Mat mask = difDynMask;
            if(mask.type() != CV_8U)
                mask.convertTo(mask, CV_8U);
            if(mask.rows != im.rows || mask.cols != im.cols)
                cv::resize(mask, mask, im.size(), 0, 0, cv::INTER_NEAREST);

            cv::Mat overlay = im.clone();
            overlay.setTo(cv::Scalar(0, 0, 255), mask);
            cv::addWeighted(overlay, 0.35, im, 0.65, 0.0, im);
        }
    }
    else if(showDIFOverlay && difOverlayMode == 2)
    {
        if(difLabel.empty())
        {
            if(difViewerHudEnable)
            {
                cv::putText(im, "NO_MASK", cv::Point(10, 28), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(0, 0, 0), 3, 8);
                cv::putText(im, "NO_MASK", cv::Point(10, 28), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(255, 255, 255), 1, 8);
            }
        }
        else
        {
            cv::Mat label = difLabel;
            if(label.type() != CV_16S)
                label.convertTo(label, CV_16S);
            if(label.rows != im.rows || label.cols != im.cols)
                cv::resize(label, label, im.size(), 0, 0, cv::INTER_NEAREST);

            double minv = 0.0, maxv = -1.0;
            cv::minMaxLoc(label, &minv, &maxv);
            const int max_id = static_cast<int>(maxv);

            const int curFid = (state == Tracking::OK) ? static_cast<int>(currentFrame.mnId) : -1;
            const int segLag = (curFid >= 0 && difSegFrameId >= 0) ? (curFid - difSegFrameId) : std::numeric_limits<int>::max();
            const bool segStale = (difSegFrameId < 0) || (curFid < 0) || (segLag > difFilterMaxLag);
            const float alphaMul = segStale ? 0.80f : 1.00f;

            const bool has_global = (difLocalToGlobalFrameId == difSegFrameId) && !difLocalToGlobal.empty();
            const bool has_state = !difTrackStates.empty();

            auto get_global_id = [&](int lid) -> int {
                if(lid < 0)
                    return -1;
                if(has_global && lid < static_cast<int>(difLocalToGlobal.size()))
                {
                    const int gid = difLocalToGlobal[static_cast<size_t>(lid)];
                    if(gid >= 0)
                        return gid;
                }
                return lid;
            };

            auto get_state = [&](int gid) -> DIFTrackState {
                const auto it = difTrackStates.find(gid);
                return (it != difTrackStates.end()) ? it->second : DIFTrackState::S;
            };

            auto get_alpha = [&](DIFTrackState st) -> float {
                float a = difViewerAlphaS;
                if(st == DIFTrackState::MS)
                    a = difViewerAlphaLock;
                else if(st == DIFTrackState::D)
                    a = difViewerAlphaD;
                return a * alphaMul;
            };

            auto state_text = [&](DIFTrackState st) -> const char* {
                if(difViewerStateStyle == 1 && st == DIFTrackState::MS)
                    return "S*";
                return DIFStateShort(st);
            };

            const bool need_stats = difViewerInstLabelEnable && (max_id >= 0);
            std::vector<DIFInstanceStat> stats;
            if(need_stats)
                stats.assign(static_cast<size_t>(max_id + 1), DIFInstanceStat());

            // Layer 1: semi-transparent fill (and collect per-instance stats for labels).
            if(difViewerInstFillEnable || need_stats)
            {
                for(int y = 0; y < label.rows; ++y)
                {
                    const int16_t* row = label.ptr<int16_t>(y);
                    cv::Vec3b* out = im.ptr<cv::Vec3b>(y);
                    for(int x = 0; x < label.cols; ++x)
                    {
                        const int lid = static_cast<int>(row[x]);
                        if(lid < 0)
                            continue;

                        const int gid = get_global_id(lid);

                        if(difViewerInstFillEnable)
                        {
                            const DIFTrackState st = has_state ? get_state(gid) : DIFTrackState::S;
                            const float a = has_state ? get_alpha(st) : (difViewerAlphaLock * alphaMul);

                            const cv::Scalar c = DIFColorFromId(gid);
                            const cv::Vec3b base = out[x];
                            out[x] = cv::Vec3b(static_cast<uint8_t>(base[0] * (1.f - a) + c[0] * a),
                                               static_cast<uint8_t>(base[1] * (1.f - a) + c[1] * a),
                                               static_cast<uint8_t>(base[2] * (1.f - a) + c[2] * a));
                        }

                        if(need_stats && lid <= max_id)
                        {
                            DIFInstanceStat& s = stats[static_cast<size_t>(lid)];
                            s.area++;
                            s.minx = std::min(s.minx, x);
                            s.miny = std::min(s.miny, y);
                            s.maxx = std::max(s.maxx, x);
                            s.maxy = std::max(s.maxy, y);
                            s.sumx += x;
                            s.sumy += y;
                        }
                    }
                }
            }

            // Layer 2: contours (only for MS/D).
            if(has_state && difViewerInstContourEnable)
            {
                cv::Mat edges = cv::Mat::zeros(label.size(), CV_8U);
                edges.colRange(0, label.cols - 1) |= (label.colRange(0, label.cols - 1) != label.colRange(1, label.cols));
                edges.rowRange(0, label.rows - 1) |= (label.rowRange(0, label.rows - 1) != label.rowRange(1, label.rows));

                cv::Mat edge_ms = cv::Mat::zeros(label.size(), CV_8U);
                cv::Mat edge_d = cv::Mat::zeros(label.size(), CV_8U);
                for(int y = 0; y < label.rows; ++y)
                {
                    const uint8_t* e = edges.ptr<uint8_t>(y);
                    const int16_t* row = label.ptr<int16_t>(y);
                    uint8_t* ms = edge_ms.ptr<uint8_t>(y);
                    uint8_t* d = edge_d.ptr<uint8_t>(y);
                    for(int x = 0; x < label.cols; ++x)
                    {
                        if(e[x] == 0)
                            continue;
                        const int lid = static_cast<int>(row[x]);
                        if(lid < 0)
                            continue;
                        const int gid = get_global_id(lid);
                        if(gid < 0)
                            continue;
                        const DIFTrackState st = get_state(gid);
                        if(st == DIFTrackState::D)
                            d[x] = 255;
                        else if(difViewerInstContourLockEnable && st == DIFTrackState::MS)
                            ms[x] = 255;
                    }
                }

                cv::Mat edge_d_thick;
                cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                cv::dilate(edge_d, edge_d_thick, kernel3); // ~2px

                for(int y = 0; y < label.rows; ++y)
                {
                    const int16_t* row = label.ptr<int16_t>(y);
                    const uint8_t* ms = edge_ms.ptr<uint8_t>(y);
                    const uint8_t* d = edge_d_thick.ptr<uint8_t>(y);
                    cv::Vec3b* out = im.ptr<cv::Vec3b>(y);
                    for(int x = 0; x < label.cols; ++x)
                    {
                        if(ms[x] == 0 && d[x] == 0)
                            continue;
                        const int lid = static_cast<int>(row[x]);
                        if(lid < 0)
                            continue;
                        const int gid = get_global_id(lid);
                        if(gid < 0)
                            continue;
                        const cv::Scalar base = DIFColorFromId(gid);
                        const cv::Scalar cc = DIFDarken(base, d[x] ? 0.55f : 0.65f);
                        out[x] = cv::Vec3b(static_cast<uint8_t>(cc[0]),
                                           static_cast<uint8_t>(cc[1]),
                                           static_cast<uint8_t>(cc[2]));
                    }
                }
            }

            // Layer 3: label inside mask (Distance Transform anchor + deterministic local search).
            if(difViewerInstLabelEnable && max_id >= 0 && !stats.empty())
            {
                const int font = cv::FONT_HERSHEY_PLAIN;
                std::vector<cv::Rect> placed;
                placed.reserve(static_cast<size_t>(max_id + 1));

                auto text_rect = [&](const cv::Point& org, const cv::Size& ts, int baseline) -> cv::Rect {
                    return cv::Rect(org.x, org.y - ts.height, ts.width, ts.height + baseline);
                };

                for(int lid = 0; lid <= max_id; ++lid)
                {
                    const DIFInstanceStat& s = stats[static_cast<size_t>(lid)];
                    if(s.area <= 0 || s.maxx < s.minx || s.maxy < s.miny)
                        continue;

                    const int gid = get_global_id(lid);
                    if(gid < 0)
                        continue;
                    const DIFTrackState st = has_state ? get_state(gid) : DIFTrackState::S;

                    const float ar = static_cast<float>(s.area) / static_cast<float>(im.rows * im.cols);

                    // Optional: filter out very small labels for all states (default: show all).
                    if(difViewerLabelMinAreaRatio > 0.0f && ar < difViewerLabelMinAreaRatio)
                        continue;

                    // Optional: filter out very small labels for S (Static) state (default: show all).
                    if(st == DIFTrackState::S && ar < difViewerLabelMinAreaRatioS)
                        continue;
                    float fontScale = 1.0f;
                    if(ar >= 0.08f)
                        fontScale = 1.25f;
                    else if(ar >= 0.02f)
                        fontScale = 1.05f;
                    else if(ar >= 0.004f)
                        fontScale = 0.85f;
                    else
                        fontScale = 0.65f;

                    const std::string text = "#" + std::to_string(gid) + " " + state_text(st);
                    const cv::Rect roi = s.Rect();
                    if(roi.empty())
                        continue;

                    cv::Mat mask_roi;
                    cv::compare(label(roi), lid, mask_roi, cv::CMP_EQ);
                    cv::Mat dist;
                    cv::distanceTransform(mask_roi, dist, cv::DIST_L2, 3);
                    cv::Point maxLoc(roi.width / 2, roi.height / 2);
                    cv::minMaxLoc(dist, nullptr, nullptr, nullptr, &maxLoc);
                    const cv::Point pstar = roi.tl() + maxLoc;

                    cv::Rect bestRect;
                    cv::Point bestOrg(0, 0);
                    bool ok = false;

                    for(int attempt = 0; attempt < 2 && !ok; ++attempt)
                    {
                        int baseline = 0;
                        const cv::Size ts = cv::getTextSize(text, font, fontScale, 2, &baseline);

                        auto try_candidate = [&](const cv::Point& anchor) -> bool {
                            if(anchor.x < 0 || anchor.y < 0 || anchor.x >= im.cols || anchor.y >= im.rows)
                                return false;
                            if(label.at<int16_t>(anchor) != lid)
                                return false;

                            const cv::Point org(anchor.x - ts.width / 2, anchor.y + ts.height / 2);
                            cv::Rect r = text_rect(org, ts, baseline);
                            if(r.x < 0 || r.y < 0 || (r.x + r.width) > im.cols || (r.y + r.height) > im.rows)
                                return false;
                            if(r.x < roi.x || r.y < roi.y || (r.x + r.width) > (roi.x + roi.width) || (r.y + r.height) > (roi.y + roi.height))
                                return false;

                            const cv::Mat mr = mask_roi(r - roi.tl());
                            const float cov = static_cast<float>(cv::countNonZero(mr)) / static_cast<float>(r.area());
                            if(cov < 0.80f)
                                return false;

                            for(const cv::Rect& pr : placed)
                            {
                                if((pr & r).area() > 0)
                                    return false;
                            }

                            bestRect = r;
                            bestOrg = org;
                            return true;
                        };

                        std::vector<cv::Point> candidates;
                        candidates.reserve(16);
                        candidates.push_back(pstar);
                        const int step = 10;
                        const std::array<cv::Point, 8> dirs = {cv::Point(1, 0), cv::Point(-1, 0), cv::Point(0, 1), cv::Point(0, -1),
                                                               cv::Point(1, 1), cv::Point(1, -1), cv::Point(-1, 1), cv::Point(-1, -1)};
                        for(int ring = 1; ring <= 2 && static_cast<int>(candidates.size()) < 16; ++ring)
                        {
                            for(const auto& d : dirs)
                            {
                                if(static_cast<int>(candidates.size()) >= 16)
                                    break;
                                candidates.push_back(pstar + d * (step * ring));
                            }
                        }

                        for(const cv::Point& q : candidates)
                        {
                            if(try_candidate(q))
                            {
                                ok = true;
                                break;
                            }
                        }

                        if(!ok)
                            fontScale = std::max(0.55f, fontScale - 0.18f);
                    }

                    if(!ok)
                    {
                        int baseline = 0;
                        const cv::Size ts = cv::getTextSize(text, font, fontScale, 2, &baseline);
                        bestOrg = cv::Point(pstar.x - ts.width / 2, pstar.y + ts.height / 2);
                    }

                    cv::putText(im, text, bestOrg + cv::Point(1, 1), font, fontScale, cv::Scalar(0, 0, 0), 3, 8);
                    cv::putText(im, text, bestOrg, font, fontScale, cv::Scalar(255, 255, 255), 1, 8);
                    if(!bestRect.empty())
                        placed.push_back(bestRect);
                }
            }

            if(segStale && difViewerHudEnable)
            {
                const char* hud = (difSegFrameId < 0) ? "NO_MASK" : "SEG_STALE";
                cv::putText(im, hud, cv::Point(10, 28), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(0, 0, 0), 3, 8);
                cv::putText(im, hud, cv::Point(10, 28), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(255, 255, 255), 1, 8);
            }
        }
    }

    //Draw
    if(state==Tracking::NOT_INITIALIZED)
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::Point2f pt1,pt2;
                if(imageScale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / imageScale;
                    pt2 = vCurrentKeys[vMatches[i]].pt / imageScale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = vCurrentKeys[vMatches[i]].pt;
                }
                cv::line(im,pt1,pt2,standardColor);
            }
        }
        for(vector<pair<cv::Point2f, cv::Point2f> >::iterator it=vTracks.begin(); it!=vTracks.end(); it++)
        {
            cv::Point2f pt1,pt2;
            if(imageScale != 1.f)
            {
                pt1 = (*it).first / imageScale;
                pt2 = (*it).second / imageScale;
            }
            else
            {
                pt1 = (*it).first;
                pt2 = (*it).second;
            }
            cv::line(im,pt1,pt2, standardColor,5);
        }

    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        int n = vCurrentKeys.size();
        const bool drawKeypoints = (!showDIFOverlay) || difViewerShowKeypoints;

        cv::Mat dynMask;
        if(showDIFOverlay && !difDynMask.empty())
        {
            dynMask = difDynMask;
            if(dynMask.type() != CV_8U)
                dynMask.convertTo(dynMask, CV_8U);
            if(dynMask.rows != im.rows || dynMask.cols != im.cols)
                cv::resize(dynMask, dynMask, im.size(), 0, 0, cv::INTER_NEAREST);
        }

        int drawn = 0;
        const int maxDraw = showDIFOverlay ? 800 : std::numeric_limits<int>::max();
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1,pt2;
                cv::Point2f point;
                if(imageScale != 1.f)
                {
                    point = vCurrentKeys[i].pt / imageScale;
                    float px = vCurrentKeys[i].pt.x / imageScale;
                    float py = vCurrentKeys[i].pt.y / imageScale;
                    pt1.x=px-r;
                    pt1.y=py-r;
                    pt2.x=px+r;
                    pt2.y=py+r;
                }
                else
                {
                    point = vCurrentKeys[i].pt;
                    pt1.x=vCurrentKeys[i].pt.x-r;
                    pt1.y=vCurrentKeys[i].pt.y-r;
                    pt2.x=vCurrentKeys[i].pt.x+r;
                    pt2.y=vCurrentKeys[i].pt.y+r;
                }

                bool allowDraw = true;
                if(!dynMask.empty())
                {
                    const cv::Point ip(static_cast<int>(point.x + 0.5f), static_cast<int>(point.y + 0.5f));
                    if(ip.x >= 0 && ip.y >= 0 && ip.x < dynMask.cols && ip.y < dynMask.rows)
                    {
                        if(dynMask.at<uint8_t>(ip) != 0)
                            allowDraw = false; // only show keypoints in static mask
                    }
                }

                // This is a match to a MapPoint in the map
                if(vbMap[i])
                {
                    mnTracked++;
                    if(drawKeypoints && allowDraw && drawn < maxDraw)
                    {
                        if(showDIFOverlay)
                            cv::circle(im, point, 1, cv::Scalar(0, 200, 0), -1);
                        else
                        {
                            cv::rectangle(im,pt1,pt2,standardColor);
                            cv::circle(im,point,2,standardColor,-1);
                        }
                        drawn++;
                    }
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    mnTrackedVO++;
                    if(drawKeypoints && allowDraw && drawn < maxDraw)
                    {
                        if(showDIFOverlay)
                            cv::circle(im, point, 1, cv::Scalar(200, 120, 0), -1);
                        else
                        {
                            cv::rectangle(im,pt1,pt2,odometryColor);
                            cv::circle(im,point,2,odometryColor,-1);
                        }
                        drawn++;
                    }
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}

cv::Mat FrameDrawer::DrawRightFrame(float imageScale)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    int state; // Tracking state

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mImRight.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeysRight;
        }
    } // destroy scoped mutex -> release mutex

    if(imageScale != 1.f)
    {
        int imWidth = im.cols / imageScale;
        int imHeight = im.rows / imageScale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::Point2f pt1,pt2;
                if(imageScale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / imageScale;
                    pt2 = vCurrentKeys[vMatches[i]].pt / imageScale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = vCurrentKeys[vMatches[i]].pt;
                }

                cv::line(im,pt1,pt2,cv::Scalar(0,255,0));
            }
        }
    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        const int n = mvCurrentKeysRight.size();
        const int Nleft = mvCurrentKeys.size();

        for(int i=0;i<n;i++)
        {
            if(vbVO[i + Nleft] || vbMap[i + Nleft])
            {
                cv::Point2f pt1,pt2;
                cv::Point2f point;
                if(imageScale != 1.f)
                {
                    point = mvCurrentKeysRight[i].pt / imageScale;
                    float px = mvCurrentKeysRight[i].pt.x / imageScale;
                    float py = mvCurrentKeysRight[i].pt.y / imageScale;
                    pt1.x=px-r;
                    pt1.y=py-r;
                    pt2.x=px+r;
                    pt2.y=py+r;
                }
                else
                {
                    point = mvCurrentKeysRight[i].pt;
                    pt1.x=mvCurrentKeysRight[i].pt.x-r;
                    pt1.y=mvCurrentKeysRight[i].pt.y-r;
                    pt2.x=mvCurrentKeysRight[i].pt.x+r;
                    pt2.y=mvCurrentKeysRight[i].pt.y+r;
                }

                // This is a match to a MapPoint in the map
                if(vbMap[i + Nleft])
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,point,2,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,point,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}

cv::Mat FrameDrawer::DrawFrameFilteredFeatures(float imageScale)
{
    cv::Mat im;
    cv::Mat imGray;
    vector<cv::KeyPoint> vCurrentKeys;
    vector<bool> vbVO, vbMap;
    int state;
    cv::Mat dynMask;
    int filtered_mode = 0;
    int filtered_max_draw = 0;
    int nTracked = 0;
    int nTrackedVO = 0;

    // Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state = mState;

        if(mState == Tracking::SYSTEM_NOT_READY)
            mState = Tracking::NO_IMAGES_YET;

        // Get the image and convert to grayscale if needed
        if(mIm.channels() == 3)
            cv::cvtColor(mIm, imGray, cv::COLOR_BGR2GRAY);
        else if(mIm.channels() == 4)
            cv::cvtColor(mIm, imGray, cv::COLOR_BGRA2GRAY);
        else
            mIm.copyTo(imGray);

        if(mState == Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
            filtered_mode = mDIFViewerFilteredFeaturesMode;
            filtered_max_draw = mDIFViewerFilteredFeaturesMaxDraw;

            // Get dynamic mask for filtering (same as original window)
            if(!mDIFDynMask.empty())
                dynMask = mDIFDynMask.clone();
        }
    } // destroy scoped mutex -> release mutex

    // Apply image scaling
    if(imageScale != 1.f)
    {
        int imWidth = imGray.cols / imageScale;
        int imHeight = imGray.rows / imageScale;
        cv::resize(imGray, imGray, cv::Size(imWidth, imHeight));

        if(!dynMask.empty())
            cv::resize(dynMask, dynMask, cv::Size(imWidth, imHeight), 0, 0, cv::INTER_NEAREST);
    }

    // Convert grayscale to BGR for drawing colored keypoints
    cvtColor(imGray, im, cv::COLOR_GRAY2BGR);

    // Draw filtered keypoints in ORB-SLAM3 style (same filtering as original window)
    if(state == Tracking::OK)
    {
        const int n_keys = static_cast<int>(vCurrentKeys.size());
        const int n_flags = std::min(static_cast<int>(vbMap.size()), static_cast<int>(vbVO.size()));
        const int n = std::min(n_keys, n_flags);

        const cv::Scalar mapColor(0, 255, 0);   // BGR: Green (Map)
        const cv::Scalar voColor(255, 0, 0);    // BGR: Blue  (VO)
        const cv::Scalar candColor(160, 160, 160); // BGR: Light gray (static candidates)
        const int r = 5;  // rectangle half-size

        const int max_draw = std::max(0, filtered_max_draw);
        const bool limit_draw = (max_draw > 0);

        auto allow_draw_at = [&](const cv::Point2f& point) -> bool
        {
            if(dynMask.empty())
                return true;
            const cv::Point ip(static_cast<int>(point.x + 0.5f), static_cast<int>(point.y + 0.5f));
            if(ip.x < 0 || ip.y < 0 || ip.x >= dynMask.cols || ip.y >= dynMask.rows)
                return true;
            return dynMask.at<uint8_t>(ip) == 0;
        };

        auto draw_tracked = [&](int idx) -> bool
        {
            const bool isMap = vbMap[idx];
            const bool isVO = vbVO[idx];
            if(!isMap && !isVO)
                return false;

            cv::Point2f point = vCurrentKeys[idx].pt;
            if(imageScale != 1.f)
                point = point / imageScale;
            if(!allow_draw_at(point))
                return false;

            const cv::Scalar color = isMap ? mapColor : voColor;
            const cv::Point2f pt1(point.x - r, point.y - r);
            const cv::Point2f pt2(point.x + r, point.y + r);
            cv::rectangle(im, pt1, pt2, color, 1);
            cv::circle(im, point, 1, color, -1);
            if(isMap)
                nTracked++;
            else
                nTrackedVO++;
            return true;
        };

        auto draw_candidate = [&](int idx) -> bool
        {
            if(vbMap[idx] || vbVO[idx])
                return false;
            cv::Point2f point = vCurrentKeys[idx].pt;
            if(imageScale != 1.f)
                point = point / imageScale;
            if(!allow_draw_at(point))
                return false;
            cv::circle(im, point, 1, candColor, -1);
            return true;
        };

        int drawn = 0;

        // Mode 0: tracked_only (only points actually used by Tracking).
        // Mode 1: static_candidates (draw tracked points + light-gray candidates outside dyn mask).
        if(filtered_mode == 1)
        {
            for(int i = 0; i < n; i++)
            {
                if(draw_tracked(i))
                {
                    drawn++;
                    if(limit_draw && drawn >= max_draw)
                        break;
                }
            }
            if(!limit_draw || drawn < max_draw)
            {
                for(int i = 0; i < n; i++)
                {
                    if(draw_candidate(i))
                    {
                        drawn++;
                        if(limit_draw && drawn >= max_draw)
                            break;
                    }
                }
            }
        }
        else
        {
            for(int i = 0; i < n; i++)
            {
                if(draw_tracked(i))
                {
                    drawn++;
                    if(limit_draw && drawn >= max_draw)
                        break;
                }
            }
        }
    }

    // Store tracked counts for status bar
    mnTracked = nTracked;
    mnTrackedVO = nTrackedVO;

    // Add status text at bottom
    cv::Mat imWithInfo;
    DrawTextInfo(im, state, imWithInfo);

    return imWithInfo;
}



void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nMaps = mpAtlas->CountMaps();
        int nKFs = mpAtlas->KeyFramesInMap();
        int nMPs = mpAtlas->MapPointsInMap();
        s << "Maps: " << nMaps << ", KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    const bool dif_show = pTracker->ShouldShowDIFOverlay();
    // Base image selection is independent of whether DIF overlay is currently enabled.
    mDIFViewerBaseRGB = pTracker->GetDIFViewerBaseRGB();
    mDIFInputIsRGB = pTracker->GetInputIsRGB();
    mDIFFilterMaxLag = pTracker->GetDIFFilterMaxLag();
    mDIFViewerShowKeypoints = pTracker->GetDIFViewerShowKeypoints();
    mDIFViewerStateStyle = pTracker->GetDIFViewerStateStyle();
    mDIFViewerInstFillEnable = pTracker->GetDIFViewerInstFillEnable();
    mDIFViewerInstContourEnable = pTracker->GetDIFViewerInstContourEnable();
    mDIFViewerInstContourLockEnable = pTracker->GetDIFViewerInstContourLockEnable();
    mDIFViewerInstLabelEnable = pTracker->GetDIFViewerInstLabelEnable();
    mDIFViewerHudEnable = pTracker->GetDIFViewerHudEnable();
    mDIFViewerAlphaS = pTracker->GetDIFViewerAlphaS();
    mDIFViewerAlphaLock = pTracker->GetDIFViewerAlphaLock();
    mDIFViewerAlphaD = pTracker->GetDIFViewerAlphaD();
    mDIFViewerLabelMinAreaRatio = pTracker->GetDIFViewerLabelMinAreaRatio();
    mDIFViewerLabelMinAreaRatioS = pTracker->GetDIFViewerLabelMinAreaRatioS();
    mDIFViewerFilteredFeaturesMode = pTracker->GetDIFViewerFilteredFeaturesMode();
    mDIFViewerFilteredFeaturesMaxDraw = pTracker->GetDIFViewerFilteredFeaturesMaxDraw();
    if(mDIFViewerBaseRGB && !pTracker->mImRGB.empty())
    {
        const cv::Mat& src = pTracker->mImRGB;
        if(src.channels() == 3 && mDIFInputIsRGB)
            cv::cvtColor(src, mIm, cv::COLOR_RGB2BGR);
        else if(src.channels() == 4 && mDIFInputIsRGB)
            cv::cvtColor(src, mIm, cv::COLOR_RGBA2BGRA);
        else
            src.copyTo(mIm);
    }
    else
    {
        pTracker->mImGray.copyTo(mIm);
    }
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    mThDepth = pTracker->mCurrentFrame.mThDepth;
    mvCurrentDepth = pTracker->mCurrentFrame.mvDepth;

    if(both){
        mvCurrentKeysRight = pTracker->mCurrentFrame.mvKeysRight;
        pTracker->mImRight.copyTo(mImRight);
        N = mvCurrentKeys.size() + mvCurrentKeysRight.size();
    }
    else{
        N = mvCurrentKeys.size();
    }

    mbShowDIFOverlay = dif_show;
    if(mbShowDIFOverlay)
    {
        mDIFOverlayMode = pTracker->GetDIFViewerOverlayMode();
        mDIFDynMask = pTracker->GetDIFDynMask().clone();
        mDIFLabelMap = pTracker->GetDIFLabelMap().clone();
        mDIFSegFrameId = pTracker->GetDIFSegFrameId();
        mDIFSegTimestamp = pTracker->GetDIFSegTimestamp();
        mDIFLocalToGlobalFrameId = pTracker->GetDIFLocalToGlobalFrameId();
        mDIFLocalToGlobal = pTracker->GetDIFLocalToGlobal();
        mDIFTrackStates = pTracker->GetDIFTrackStates();
        mDIFDynMaskFrameId = pTracker->GetDIFDynMaskFrameId();
        mDIFDynMaskTimestamp = pTracker->GetDIFDynMaskTimestamp();
        mDIFSegElapsedMs = pTracker->GetDIFSegElapsedMs();
    }
    else
    {
        mDIFOverlayMode = 0;
        mDIFDynMask.release();
        mDIFLabelMap.release();
        mDIFSegFrameId = -1;
        mDIFSegTimestamp = -1.0;
        mDIFLocalToGlobalFrameId = -1;
        mDIFLocalToGlobal.clear();
        mDIFTrackStates.clear();
        mDIFDynMaskFrameId = -1;
        mDIFDynMaskTimestamp = -1.0;
        mDIFSegElapsedMs = 0.0;
    }

    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;

    //Variables for the new visualization
    mCurrentFrame = pTracker->mCurrentFrame;
    mLastFrameId = static_cast<int>(pTracker->mCurrentFrame.mnId);
    mmProjectPoints = mCurrentFrame.mmProjectPoints;
    mmMatchedInImage.clear();

    mvpLocalMap = pTracker->GetLocalMapMPS();
    mvMatchedKeys.clear();
    mvMatchedKeys.reserve(N);
    mvpMatchedMPs.clear();
    mvpMatchedMPs.reserve(N);
    mvOutlierKeys.clear();
    mvOutlierKeys.reserve(N);
    mvpOutlierMPs.clear();
    mvpOutlierMPs.reserve(N);

    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        for(int i=0;i<N;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;

                    mmMatchedInImage[pMP->mnId] = mvCurrentKeys[i].pt;
                }
                else
                {
                    mvpOutlierMPs.push_back(pMP);
                    mvOutlierKeys.push_back(mvCurrentKeys[i]);
                }
            }
        }

    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
}

int FrameDrawer::GetLastFrameId()
{
    unique_lock<mutex> lock(mMutex);
    return mLastFrameId;
}

} //namespace ORB_SLAM
