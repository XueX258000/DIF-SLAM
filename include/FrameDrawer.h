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


#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "MapPoint.h"
#include "Atlas.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<mutex>
#include <unordered_set>
#include <unordered_map>


namespace ORB_SLAM3
{

class Tracking;
class Viewer;
enum class DIFTrackState : int;

class FrameDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FrameDrawer(Atlas* pAtlas);

    // Update info from the last processed frame.
    void Update(Tracking *pTracker);

    // Draw last processed frame.
    cv::Mat DrawFrame(float imageScale=1.f);
    cv::Mat DrawRightFrame(float imageScale=1.f);

    // Draw frame with only filtered features (post-segmentation)
    cv::Mat DrawFrameFilteredFeatures(float imageScale=1.f);
    int GetLastFrameId();

    bool both;

protected:

    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // Info of the frame to be drawn
    cv::Mat mIm, mImRight;
    int N;
    vector<cv::KeyPoint> mvCurrentKeys,mvCurrentKeysRight;
    vector<bool> mvbMap, mvbVO;
    bool mbOnlyTracking;
    int mnTracked, mnTrackedVO;
    vector<cv::KeyPoint> mvIniKeys;
    vector<int> mvIniMatches;
    int mState;
    int mLastFrameId;
    std::vector<float> mvCurrentDepth;
    float mThDepth;

    Atlas* mpAtlas;

    std::mutex mMutex;
    vector<pair<cv::Point2f, cv::Point2f> > mvTracks;

    Frame mCurrentFrame;
    vector<MapPoint*> mvpLocalMap;
    vector<cv::KeyPoint> mvMatchedKeys;
    vector<MapPoint*> mvpMatchedMPs;
    vector<cv::KeyPoint> mvOutlierKeys;
    vector<MapPoint*> mvpOutlierMPs;

    map<long unsigned int, cv::Point2f> mmProjectPoints;
    map<long unsigned int, cv::Point2f> mmMatchedInImage;

    bool mbShowDIFOverlay = false;
    int mDIFOverlayMode = 0; // 0: off, 1: dyn mask, 2: instance colors
    bool mDIFViewerBaseRGB = false; // if true, use Tracking::mImRGB as base for visualization
    bool mDIFInputIsRGB = false;    // Tracking input color order (Camera.RGB)
    int mDIFFilterMaxLag = 2;       // frames; for SEG_STALE/NO_MASK HUD & alpha tweak
    bool mDIFViewerShowKeypoints = true; // if true, show (weak) keypoints in static mask
    int mDIFViewerStateStyle = 1;
    bool mDIFViewerInstFillEnable = true;
    bool mDIFViewerInstContourEnable = true;
    bool mDIFViewerInstContourLockEnable = true;
    bool mDIFViewerInstLabelEnable = true;
    bool mDIFViewerHudEnable = true;
    float mDIFViewerAlphaS = 0.16f;
    float mDIFViewerAlphaLock = 0.22f;
    float mDIFViewerAlphaD = 0.35f;
    float mDIFViewerLabelMinAreaRatio = 0.0f;
    float mDIFViewerLabelMinAreaRatioS = 0.0f;
    int mDIFViewerFilteredFeaturesMode = 0;
    int mDIFViewerFilteredFeaturesMaxDraw = 300;
    int mDIFSegFrameId = -1;
    double mDIFSegTimestamp = -1.0;
    double mDIFSegElapsedMs = 0.0;
    cv::Mat mDIFLabelMap;  // CV_16S ([-1, N-1])
    int mDIFLocalToGlobalFrameId = -1;
    std::vector<int> mDIFLocalToGlobal; // local_id -> global track id (Module B)
    std::unordered_map<int, DIFTrackState> mDIFTrackStates; // global_id -> state_hat (Module C)

    int mDIFDynMaskFrameId = -1;
    double mDIFDynMaskTimestamp = -1.0;
    cv::Mat mDIFDynMask;   // CV_8U (0/255)

};

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H
