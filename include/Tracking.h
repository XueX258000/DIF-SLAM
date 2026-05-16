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


#ifndef TRACKING_H
#define TRACKING_H

#include <limits>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "MapDrawer.h"
#include "System.h"
#include "ImuTypes.h"
#include "Settings.h"

#include "GeometricCamera.h"
#include "DIF/SegmentationWorker.h"
#include "DIF/InstanceTracker.h"
#include "DIF/DynamicStateEstimator.h"
#include "DIF/DenseMapping.h"

#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <memory>
#include <utility>

namespace ORB_SLAM3
{

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;

class Tracking
{  

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Atlas* pAtlas,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq=std::string());

    ~Tracking();

    // Parse the config file
    bool ParseCamParamFile(cv::FileStorage &fSettings);
    bool ParseORBParamFile(cv::FileStorage &fSettings);
    bool ParseIMUParamFile(cv::FileStorage &fSettings);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    Sophus::SE3f GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename);
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);

    void GrabImuData(const IMU::Point &imuMeasurement);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
    void SetStepByStep(bool bSet);
    bool GetStepByStep();

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);
    KeyFrame* GetLastKeyFrame()
    {
        return mpLastKeyFrame;
    }

    void CreateMapInAtlas();
    //std::mutex mMutexTracks;

    //--
    void NewDataset();
    int GetNumberDataset();
    int GetMatchesInliers();

    //DEBUG
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder="");
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap);

    float GetImageScale();

#ifdef REGISTER_LOOP
    void RequestStop();
    bool isStopped();
    void Release();
    bool stopRequested();
#endif

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        RECENTLY_LOST=3,
        LOST=4,
        OK_KLT=5
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;

    cv::Mat mImGray;
    // Previous frame grayscale image (for LK rigid-flow residuals).
    cv::Mat mImGrayPrev;
    cv::Mat mImRGB;
    cv::Mat mImDepth;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Sophus::SE3f> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // frames with estimated pose
    int mTrackedFr;
    bool mbStep;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset(bool bLocMap = false);
    void ResetActiveMap(bool bLocMap = false);

    float mMeanTrack;
    bool mbInitWith3KFs;
    double t0; // time-stamp of first read frame
    double t0vis; // time-stamp of first inserted keyframe
    double t0IMU; // time-stamp of IMU initialization
    bool mFastInit = false;


    vector<MapPoint*> GetLocalMapMPS();

    bool mbWriteStats;

#ifdef REGISTER_TIMES
    void LocalMapStats2File();
    void TrackStats2File();
    void PrintTimeStats();

    vector<double> vdRectStereo_ms;
    vector<double> vdResizeImage_ms;
    vector<double> vdORBExtract_ms;
    vector<double> vdStereoMatch_ms;
    vector<double> vdIMUInteg_ms;
    vector<double> vdPosePred_ms;
    vector<double> vdLMTrack_ms;
    vector<double> vdNewKF_ms;
    vector<double> vdTrackTotal_ms;
#endif

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    //void CreateNewMapPoints();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();
    bool PredictStateIMU();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // Perform preintegration from last frame
    void PreintegrateIMU();

    // Reset IMU biases and compute frame velocity
    void ResetFrameIMU();

    bool mbMapUpdated;

    // Imu preintegration from last frame
    IMU::Preintegrated *mpImuPreintegratedFromLastKF;

    // Queue of IMU measurements between frames
    std::list<IMU::Point> mlQueueImuData;

    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> mvImuFromLastFrame;
    std::mutex mMutexImuQueue;

    // Imu calibration parameters
    IMU::Calib *mpImuCalib;

    // Last Bias Estimation (at keyframe creation)
    IMU::Bias mLastBias;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    bool mbReadyToInitializate;
    bool mbSetInit;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    bool bStepByStep;

    //Atlas
    Atlas* mpAtlas;

    //Calibration matrix
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    cv::Mat mDistCoef;
    float mbf;
    float mImageScale;

    float mImuFreq;
    double mImuPer;
    bool mInsertKFsLost;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    int mnFirstImuFrameId;
    int mnFramesToResetIMU;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    double mTimeStampLost;
    double time_recently_lost;

    unsigned int mnFirstFrameId;
    unsigned int mnInitialFrameId;
    unsigned int mnLastInitFrameId;

    bool mbCreatedMap;

    //Motion Model
    bool mbVelocity{false};
    Sophus::SE3f mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;

    //int nMapChangeIndex;

    int mnNumDataset;

    ofstream f_track_stats;

    ofstream f_track_times;
    double mTime_PreIntIMU;
    double mTime_PosePred;
    double mTime_LocalMapTrack;
    double mTime_NewKF_Dec;

    GeometricCamera* mpCamera, *mpCamera2;

    int initID, lastID;

    Sophus::SE3f mTlr;

    void newParameterLoader(Settings* settings);

    struct DIFFrameCacheEntry;

    bool ParseDIFParamFile(cv::FileStorage &fSettings);
    void UpdateDIFSegmentation();
    void ApplyDIFMaskToMatchedMapPoints(Frame& frame);
    bool IsInDIFMask(const cv::Mat& mask, const cv::Point2f& pt) const;
    void UpdateDIFDebugMaskFromLabelMap(const SegmentationResult& seg);
    void PushDIFFrameCache();
    void StoreDIFPendingSegmentation(const SegmentationResult& seg);
    void ProcessDIFPendingSegmentations();
    void UpdateDIFTrackVobsFromFeature3D(int frame_id,
                                         const cv::Mat& label_map,
                                         const DIFFrameCacheEntry& fe,
                                         const std::vector<int>& local_to_global,
                                         bool allow_3d_update);
    int GetDIFGlobalTrackIdAt(const cv::Point2f& pt) const;
    void StoreDIFSegCacheLabelMap(int frame_id, const cv::Mat& label_map);
    void StoreDIFSegCacheLocalToGlobal(int frame_id, const std::vector<int>& local_to_global);
    void UpdateDIFSuppressedTrackIds();
    void UpdateDIFForbiddenForMapTrackIds();
    void UpdateDIFTrackLockInfoSnapshot();
    cv::Mat BuildDIFDynamicMaskFromTracks(int height, int width, double timestamp) const;
    cv::Mat BuildDIFForbidMaskForDense(int height, int width, double timestamp,
                                       bool has_pose,
                                       const Sophus::SE3f& Tcw) const;
    const cv::Mat& GetDIFDynamicMaskPredForCurrentFrame();
    const cv::Mat& GetDIFForbidMaskPredForCurrentFrame();
    void InvalidateDIFPredMasks();
    void ResetDIFRuntimeState(bool reset_instance_tracker);
    void MaybeDIFSecondPassPoseOptimization();
    int CountDIFFrameInliers(const Frame& frame) const;
    void MaybeEnqueueDenseMappingFromKeyFrame(KeyFrame* pKF);
    void MaybeFlushPendingDenseKeyFrame(int frame_id);
    void MaybeEnqueueDenseMappingFromSegFrame(int frame_id,
                                              double timestamp,
                                              const cv::Mat& label_map,
                                              const DIFFrameCacheEntry& fe,
                                              const std::vector<int>& local_to_global);
    bool GetDenseSegmentationForFrame(int frame_id,
                                      cv::Mat& out_label_map,
                                      std::vector<int>& out_local_to_global,
                                      int& out_seg_frame_id) const;
    bool BuildDensePacketFromFrame(int frame_id,
                                   double timestamp,
                                   const cv::Mat& rgb,
                                   const cv::Mat& depth,
                                   const Sophus::SE3f& Tcw,
                                   KeyFrame* pKF,
                                   DenseKFPacket& out_packet);
    void MaybeRebuildDenseMappingAfterLoop();
    void SaveDIFBinaryMaskForCurrentFrame();
    void SaveFlowVisForCurrentFrame();

    SegmentationConfig mDIFSegCfg;
    std::unique_ptr<SegmentationWorker> mpDIFSegWorker;
    cv::Mat mDIFDynMaskLast; // CV_8U (0/255), used for filtering
    cv::Mat mDIFLabelMapLast; // CV_16S, value range [-1, N-1], resized to current frame
    bool mDIFBinaryMaskVisEnable = true; // Save per-frame binary mask to Output/binary_mask
    bool mDIFFlowVisEnable = false; // Save per-frame sparse optical flow visualization to Output/flow_vis
    int mDIFFlowVisStep = 20; // grid step (pixels)
    float mDIFFlowVisThreshold = 1.0f; // min displacement (px) to draw
    float mDIFFlowVisScale = 3.0f; // scale factor for arrow length
    float mDIFFlowVisMaxLen = 30.0f; // max arrow length (px), <=0 disables clipping
    bool mDIFFilterEnable = true;
    int mDIFFilterMaxLag = 2; // frames; skip filtering if seg result too old
    int mDIFDebugMaskMode = 0; // 0: off, 1: all instances, 2: largest instance
    int mDIFDebugDilate = 0;   // kernel size (0: off)
    int mDIFViewerOverlayMode = 2; // 0: off, 1: dyn mask, 2: instance colors (label_map)
    bool mDIFViewerBaseRGB = true; // 0: use gray base, 1: use RGB base (visualization)
    bool mDIFViewerShowKeypoints = true; // if true, show (weak) keypoints in static mask
    // Viewer style for instance overlay (mode=2):
    // - 0: legacy S/MS/D (MS shows as "MS")
    // - 1: two-state + map_lock (map_lock shows as "S*")
    int mDIFViewerStateStyle = 1;
    bool mDIFViewerInstFillEnable = true;
    bool mDIFViewerInstContourEnable = true;
    bool mDIFViewerInstContourLockEnable = true;
    bool mDIFViewerInstLabelEnable = true;
    bool mDIFViewerHudEnable = true;
    float mDIFViewerAlphaS = 0.16f;
    float mDIFViewerAlphaLock = 0.22f;
    float mDIFViewerAlphaD = 0.35f;
    float mDIFViewerLabelMinAreaRatio = 0.0f; // 0: show all labels
    float mDIFViewerLabelMinAreaRatioS = 0.0f; // 0: show all static labels
    // Tracking Features (Filtered) window draw mode:
    // - 0: tracked_only (only vbMap/vbVO, i.e., points actually used by Tracking)
    // - 1: static_candidates (all keypoints outside dyn mask; tracked points highlighted)
    int mDIFViewerFilteredFeaturesMode = 0;
    int mDIFViewerFilteredFeaturesMaxDraw = 300; // 0: unlimited
    bool mDIFSync = false; // if true, wait for seg result for submitted frame (visualization-friendly)
    int mDIFSyncTimeoutMs = 0; // 0: no wait
    bool mDIFInitialSync = false; // if true, wait only for the first submitted segmentation to avoid cold-start lag
    int mDIFInitialSyncTimeoutMs = 0; // 0: no wait
    bool mDIFInitialSyncDone = false;
    bool mDIFSecondPassEnable = false; // if true, run a second PoseOptimization using current M_t^{dyn}
    bool mDIFSecondPassEnableIMU = false; // if true, allow second-pass when IMU is initialized (use inertial optimization)
    int mDIFSecondPassMinInliers = 30; // skip if first-pass inliers are too low (avoid wasting compute)
    float mDIFSecondPassIouMin = 0.50f; // trigger when IoU(M_t, M_{t-1}) < this threshold
    float mDIFSecondPassDynRatioDelta = 0.10f; // trigger when |dyn_ratio_t - dyn_ratio_{t-1}| > this threshold
    int mDIFSecondPassMaxLag = 1; // only compare with previous mask if lag <= this (frames)
    int mDIFLastSegFrameId = -1;
    double mDIFLastSegTimestamp = -1.0;
    int mDIFLastDynMaskFrameId = -1;
    double mDIFLastDynMaskTimestamp = -1.0;
    int mDIFLastPrintSegFrameId = -1;
    int mDIFLastPrintBFrameId = -1;
    double mDIFLastSegElapsedMs = 0.0;

    // Module B: global instance tracking
    DIFInstanceTrackerConfig mDIFTrackCfg;
    std::unique_ptr<DIFInstanceTracker> mpDIFInstanceTracker;
    // Monotonic frame_id of the last successful Module-B update. Used to prevent out-of-order segmentation updates
    // from corrupting track continuity (which would cause global instance IDs to change/swap).
    int mDIFLastBUpdateFrameId = -1;
    int mDIFLocalToGlobalFrameId = -1;
    std::vector<int> mDIFLocalToGlobal; // local_id -> global track id

    // Module C: HMM + circuit breaker + hysteresis
    DIFDynamicStateConfig mDIFStateCfg;
    std::unique_ptr<DIFDynamicStateEstimator> mpDIFStateEstimator;
    std::unordered_set<int> mDIFSuppressedTrackIds;
    mutable std::mutex mMutexDIFSuppressed;
    std::unordered_set<int> mDIFForbiddenForMapTrackIds;
    mutable std::mutex mMutexDIFForbiddenForMap;
    struct DIFTrackLockInfo
    {
        bool map_lock = false;
        float w_lock = 1.0f;
    };
    std::unordered_map<int, DIFTrackLockInfo> mDIFTrackLockInfo;
    mutable std::mutex mMutexDIFTrackLockInfo;
    int mDIFFilterLastLoggedFrameId = -1;
    cv::Mat mDIFPrevDynMask;
    int mDIFPrevDynMaskFrameId = -1;

    // Per-frame predicted masks (built from tracks' last masks with optional warp/age gating).
    // These masks are computed on-demand and invalidated whenever tracks are updated (segmentation-update frames).
    cv::Mat mDIFDynMaskPred;
    int mDIFDynMaskPredFrameId = -1;
    double mDIFDynMaskPredTimestamp = -1.0;
    cv::Mat mDIFForbidMaskPred;
    int mDIFForbidMaskPredFrameId = -1;
    double mDIFForbidMaskPredTimestamp = -1.0;

    struct DIFFrameCacheEntry
    {
        int frame_id = -1;
        double timestamp = -1.0;
        Sophus::SE3f Tcw;
        bool has_pose = false;
        cv::Mat rgb;   // CV_8UC3
        cv::Mat depth; // CV_32F
        std::vector<cv::Point2f> keypoints_uv;
        std::vector<cv::Point2f> keypoints_un_uv;

        // Inlier keypoint matches to previous frame (prev_idx, cur_idx). Used by Module-B v_obs (feature 3D residual).
        int match_prev_frame_id = -1;
        std::vector<std::pair<int, int>> matches_prev;

        // Pose quality (for Module B 8.2 gate)
        int inliers = 0;
        bool has_r_bg = false;
        float r_bg = std::numeric_limits<float>::infinity(); // px
    };

    int mDIFFrameCacheMaxSize = 30;
    std::deque<DIFFrameCacheEntry> mDIFFrameCache;
    std::unordered_map<int, SegmentationResult> mDIFPendingSegByFrameId;
    struct DIFSegCacheEntry
    {
        cv::Mat label_map;               // CV_16S, segmentation output size for that frame_id
        std::vector<int> local_to_global; // local_id -> global track id (Module B), may be empty if not available
    };
    int mDIFSegCacheMaxSize = 200;
    std::deque<int> mDIFSegCacheOrder;
    std::unordered_map<int, DIFSegCacheEntry> mDIFSegCacheByFrameId;
    mutable std::mutex mMutexDIFSegCache;

    // Dense mapping (optional)
    DenseMappingConfig mDenseCfg;
    std::unique_ptr<DenseMapping> mpDenseMapping;
    int mDenseLastMapChangeIdx = -1;
    struct DensePendingKeyFrame
    {
        KeyFrame* pKF = nullptr;
        int frame_id = -1;
        double timestamp = -1.0;
        cv::Mat rgb;
        cv::Mat depth;
    };
    std::unordered_map<int, DensePendingKeyFrame> mDensePendingKeyFrames;

#ifdef REGISTER_LOOP
    bool Stop();

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;
#endif

public:
    cv::Mat mImRight;

    bool IsDIFEnabled() const { return mDIFSegCfg.enable; }
    bool IsDIFFilterEnabled() const { return mDIFFilterEnable; }
    bool IsDIFTrackSuppressed(int global_track_id) const;
    bool IsDIFTrackForbiddenForMap(int global_track_id) const;
    // Snapshot query (thread-safe): current map_lock and w_lock for a global track id.
    // Returns false if the track id is unknown or Module C is disabled.
    bool GetDIFTrackLockInfo(int global_track_id, bool& out_map_lock, float& out_w_lock) const;
    bool ShouldShowDIFOverlay() const { return mDIFSegCfg.enable && (mDIFViewerOverlayMode > 0); }
    int GetDIFViewerOverlayMode() const { return mDIFViewerOverlayMode; }
    bool GetDIFViewerBaseRGB() const { return mDIFViewerBaseRGB; }
    int GetDIFViewerStateStyle() const { return mDIFViewerStateStyle; }
    bool GetDIFViewerInstFillEnable() const { return mDIFViewerInstFillEnable; }
    bool GetDIFViewerInstContourEnable() const { return mDIFViewerInstContourEnable; }
    bool GetDIFViewerInstContourLockEnable() const { return mDIFViewerInstContourLockEnable; }
    bool GetDIFViewerInstLabelEnable() const { return mDIFViewerInstLabelEnable; }
    bool GetDIFViewerHudEnable() const { return mDIFViewerHudEnable; }
    float GetDIFViewerAlphaS() const { return mDIFViewerAlphaS; }
    float GetDIFViewerAlphaLock() const { return mDIFViewerAlphaLock; }
    float GetDIFViewerAlphaD() const { return mDIFViewerAlphaD; }
    float GetDIFViewerLabelMinAreaRatio() const { return mDIFViewerLabelMinAreaRatio; }
    float GetDIFViewerLabelMinAreaRatioS() const { return mDIFViewerLabelMinAreaRatioS; }
    int GetDIFViewerFilteredFeaturesMode() const { return mDIFViewerFilteredFeaturesMode; }
    int GetDIFViewerFilteredFeaturesMaxDraw() const { return mDIFViewerFilteredFeaturesMaxDraw; }

    // Start DIF SegmentationWorker thread (delayed initialization to avoid race condition)
    void StartDIFSegmentationWorker();
    void StartDenseMappingWorker();
    void StopDenseMappingWorker();
    bool IsDenseMappingEnabled() const { return (mDenseCfg.enable && mpDenseMapping); }
    bool SaveDensePointCloud(const std::string& path, bool use_latest_pose = true);
    bool GetDIFViewerShowKeypoints() const { return mDIFViewerShowKeypoints; }
    int GetDIFFilterMaxLag() const { return mDIFFilterMaxLag; }
    bool GetInputIsRGB() const { return mbRGB; }
    const cv::Mat& GetDIFDynMask() const { return mDIFDynMaskLast; }
    const cv::Mat& GetDIFLabelMap() const { return mDIFLabelMapLast; }
    int GetDIFMaskFrameId() const { return mDIFLastDynMaskFrameId; }
    double GetDIFMaskTimestamp() const { return mDIFLastDynMaskTimestamp; }
    double GetDIFSegElapsedMs() const { return mDIFLastSegElapsedMs; }
    int GetDIFSegFrameId() const { return mDIFLastSegFrameId; }
    double GetDIFSegTimestamp() const { return mDIFLastSegTimestamp; }
    int GetDIFDynMaskFrameId() const { return mDIFLastDynMaskFrameId; }
    double GetDIFDynMaskTimestamp() const { return mDIFLastDynMaskTimestamp; }

    int GetDIFLocalToGlobalFrameId() const { return mDIFLocalToGlobalFrameId; }
    const std::vector<int>& GetDIFLocalToGlobal() const { return mDIFLocalToGlobal; }

    // Snapshot of current track states (global_id -> state_hat). Empty if Module B/C not enabled.
    std::unordered_map<int, DIFTrackState> GetDIFTrackStates() const;

    // Query global track id for a keypoint in a specific frame_id (used by LocalMapping for triangulated points).
    // Returns -1 if segmentation/mapping is unavailable.
    int GetDIFGlobalTrackIdAtFrameId(int frame_id, const cv::Point2f& pt) const;

    // True if both label_map and local_to_global are available in the DIF seg cache for the given frame_id.
    bool HasDIFSegMappingForFrameId(int frame_id) const;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
