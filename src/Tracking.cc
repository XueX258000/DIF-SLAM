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


#include "Tracking.h"

#include <cmath>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"

#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstdlib>

#include <cerrno>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

#include <sys/stat.h>
#include <sys/types.h>

#include <mutex>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>


using namespace std;

namespace {

bool EnsureDir(const std::string& path)
{
    struct stat st;
    if(stat(path.c_str(), &st) == 0)
        return S_ISDIR(st.st_mode);
    if(mkdir(path.c_str(), 0755) == 0)
        return true;
    return errno == EEXIST;
}

std::string BuildFramePath(const std::string& dir, int frame_id)
{
    std::ostringstream oss;
    oss << dir << "/frame_" << std::setw(6) << std::setfill('0') << frame_id << ".png";
    return oss.str();
}

std::string BuildTmpFramePath(const std::string& dir, int frame_id)
{
    std::ostringstream oss;
    oss << dir << "/.tmp_frame_" << std::setw(6) << std::setfill('0') << frame_id << ".png";
    return oss.str();
}

} // namespace

namespace ORB_SLAM3
{


Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL))
{
    // Load camera parameters from settings file
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;
        if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO || sensor==System::IMU_RGBD)
        {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if(!b_parse_imu)
            {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
	    }

    {
        cv::FileStorage fDIF(strSettingPath, cv::FileStorage::READ);
        if(fDIF.isOpened())
        {
            ParseDIFParamFile(fDIF);
        }
    }

    if(mDIFSegCfg.enable)
    {
        std::cout << "[DIF] enabled, server=" << mDIFSegCfg.server_script
                  << " weights=" << mDIFSegCfg.weights
                  << " device=" << mDIFSegCfg.device
                  << " every_n_frames=" << mDIFSegCfg.every_n_frames
                  << " imgsz=" << mDIFSegCfg.imgsz
                  << " sync=" << (mDIFSync ? 1 : 0)
                  << " half=" << (mDIFSegCfg.half ? 1 : 0)
                  << std::endl;
        // Create SegmentationWorker but DON'T start the thread yet
        // Thread will be started after all system threads are initialized
        // to avoid multi-thread race condition during startup
        mpDIFSegWorker.reset(new SegmentationWorker(mDIFSegCfg));

        if(mDIFTrackCfg.enable)
        {
            mpDIFInstanceTracker.reset(new DIFInstanceTracker(mDIFTrackCfg));
            std::cout << "[DIF-B] enabled, area_min=" << mDIFTrackCfg.area_min
                      << " n_feat_min=" << mDIFTrackCfg.n_feat_min
                      << " stride=" << mDIFTrackCfg.sample_stride
                      << " max_miss=" << mDIFTrackCfg.max_miss
                      << " pose_inliers_min=" << mDIFTrackCfg.pose_inliers_min
                      << " pose_rbg_max=" << mDIFTrackCfg.pose_rbg_max
                      << std::endl;
        }

        if(mDIFStateCfg.enable)
        {
            mpDIFStateEstimator.reset(new DIFDynamicStateEstimator(mDIFStateCfg));
            std::cout << "[DIF-C] enabled, tau_up=" << mDIFStateCfg.tau_up
                      << " tau_down=" << mDIFStateCfg.tau_down
                      << " tau_bg=" << mDIFStateCfg.tau_bg
                      << " flow_mode=" << (mDIFStateCfg.flow_mode == 1 ? "lk" : "desc")
                      << " state_mode=" << mDIFStateCfg.state_mode
                      << " map_lock=" << (mDIFStateCfg.map_lock_enable ? 1 : 0)
                      << " map_confirm=" << mDIFStateCfg.map_confirm_N
                      << " map_cooldown=" << mDIFStateCfg.map_cooldown_N
                      << " dyn_mask_ever=" << (mDIFStateCfg.dyn_mask_include_ever_dynamic ? 1 : 0)
                      << std::endl;
        }

        if(mDenseCfg.enable)
        {
            const bool rgbd_ok = (mSensor == System::RGBD || mSensor == System::IMU_RGBD);
            const bool dif_ready = (mDIFTrackCfg.enable && mDIFStateCfg.enable);
            if(!rgbd_ok || !dif_ready)
            {
                std::cout << "[DENSE] disabled (requires RGBD + DIF.B + DIF.C)" << std::endl;
                mDenseCfg.enable = false;
            }
            else
            {
                mpDenseMapping.reset(new DenseMapping(mDenseCfg));
                std::cout << "[DENSE] enabled, integrate_on=" << mDenseCfg.integrate_on
                          << " voxel=" << mDenseCfg.voxel_size_m
                          << " stride=" << mDenseCfg.stride
                          << " confirm=" << mDenseCfg.n_dense_confirm
                          << " exact_label=" << (mDenseCfg.require_exact_label_map ? 1 : 0)
                          << std::endl;
            }
        }
    }

    initID = 0; lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for(GeometricCamera* pCam : vpCams)
    {
        std::cout << "Camera " << pCam->GetId();
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            std::cout << " is pinhole" << std::endl;
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            std::cout << " is fisheye" << std::endl;
        }
        else
        {
            std::cout << " is unknown" << std::endl;
        }
    }

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif
}

std::unordered_map<int, DIFTrackState> Tracking::GetDIFTrackStates() const
{
    std::unordered_map<int, DIFTrackState> out;
    if(!mDIFSegCfg.enable || !mpDIFInstanceTracker || !mDIFStateCfg.enable)
        return out;
    const auto& tracks = mpDIFInstanceTracker->GetTracks();
    out.reserve(tracks.size());
    for(const auto& kv : tracks)
    {
        const DIFTrack& tr = kv.second;
        // Viewer-friendly display: in two-state+map_lock mode, show map_lock as "MS" (buffer/locked) to keep
        // visualization semantics consistent with docs/参考资料/可视化.md while preserving two-state logic.
        DIFTrackState st = tr.state_hat;
        if(st != DIFTrackState::D && tr.map_lock)
            st = DIFTrackState::MS;
        out.emplace(kv.first, st);
    }
    return out;
}

void Tracking::StartDIFSegmentationWorker()
{
    if(mpDIFSegWorker)
    {
        mpDIFSegWorker->Start();
        std::cout << "[DIF] SegmentationWorker thread started (delayed init)" << std::endl;
    }
}

void Tracking::StartDenseMappingWorker()
{
    if(mpDenseMapping)
    {
        mpDenseMapping->Start();
        std::cout << "[DENSE] DenseMapping thread started" << std::endl;
    }
}

void Tracking::StopDenseMappingWorker()
{
    if(mpDenseMapping)
        mpDenseMapping->Stop();
}

bool Tracking::IsDIFTrackSuppressed(int global_track_id) const
{
    if(global_track_id < 0)
        return false;
    std::lock_guard<std::mutex> lk(mMutexDIFSuppressed);
    return mDIFSuppressedTrackIds.find(global_track_id) != mDIFSuppressedTrackIds.end();
}

bool Tracking::IsDIFTrackForbiddenForMap(int global_track_id) const
{
    if(global_track_id < 0)
        return false;
    if(!mDIFStateCfg.enable)
        return false;
    std::lock_guard<std::mutex> lk(mMutexDIFForbiddenForMap);
    return mDIFForbiddenForMapTrackIds.find(global_track_id) != mDIFForbiddenForMapTrackIds.end();
}

bool Tracking::GetDIFTrackLockInfo(int global_track_id, bool& out_map_lock, float& out_w_lock) const
{
    out_map_lock = false;
    out_w_lock = 1.0f;
    if(global_track_id < 0)
        return false;
    if(!mDIFStateCfg.enable)
        return false;
    std::lock_guard<std::mutex> lk(mMutexDIFTrackLockInfo);
    const auto it = mDIFTrackLockInfo.find(global_track_id);
    if(it == mDIFTrackLockInfo.end())
        return false;
    out_map_lock = it->second.map_lock;
    out_w_lock = it->second.w_lock;
    return true;
}

bool Tracking::SaveDensePointCloud(const std::string& path, bool use_latest_pose)
{
    if(!mpDenseMapping || !mDenseCfg.enable)
        return false;
    return mpDenseMapping->SaveDensePointCloud(path, use_latest_pose);
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();

}

#endif

Tracking::~Tracking()
{
    if(mpDIFSegWorker)
        mpDIFSegWorker->Stop();
    if(mpDenseMapping)
        mpDenseMapping->Stop();

}

void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if(settings->needToUndistort()){
        mDistCoef = settings->camera1DistortionCoef();
    }
    else{
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    //TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    if((mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD) &&
        settings->cameraType() == Settings::KannalaBrandt){
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr = settings->Tlr();

        mpFrameDrawer->both = true;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD ){
        mbf = settings->bf();
        mThDepth = settings->b() * settings->thDepth();
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD){
        mDepthMapFactor = settings->depthMapFactor();
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    //ORB parameters
    int nFeatures = settings->nFeatures();
    int nLevels = settings->nLevels();
    int fIniThFAST = settings->initThFAST();
    int fMinThFAST = settings->minThFAST();
    float fScaleFactor = settings->scaleFactor();

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    mImuPer = 0.001; //1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        if(mImageScale != 1.f)
        {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- Image scale: " << mImageScale << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if(mDistCoef.rows==5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

        mK_.setIdentity();
        mK_(0,0) = fx;
        mK_(1,1) = fy;
        mK_(0,2) = cx;
        mK_(1,2) = cy;
    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(!b_miss_params)
        {
            if(mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;

            mK_.setIdentity();
            mK_(0,0) = fx;
            mK_(1,1) = fy;
            mK_(0,2) = cx;
            mK_(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if(!node.empty())
            {
                cvTlr = node.mat();
                if(cvTlr.rows != 3 || cvTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                if(mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << cvTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if(b_miss_params)
        {
            return false;
        }

    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD )
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
            if(mImageScale != 1.f)
            {
                mbf *= mImageScale;
            }
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }


    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    cout << endl;
    cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        cout << "Do not insert keyframes when lost visual tracking " << endl;



    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if(mFastInit)
        cout << "Fast IMU initialization. Acceleration is not checked \n";

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    cout << endl;
    cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

bool Tracking::ParseDIFParamFile(cv::FileStorage &fSettings)
{
    cv::FileNode node = fSettings["DIF.enable"];
    if(node.empty())
        return true;

    if(node.isInt())
        mDIFSegCfg.enable = (node.operator int() != 0);
    else if(node.isString())
        mDIFSegCfg.enable = (node.string() == "1" || node.string() == "true" || node.string() == "True");
    else
        mDIFSegCfg.enable = false;

    // Flow visualization can be used independently from DIF enable (debug/analysis utility).
    node = fSettings["DIF.flow_vis_enable"];
    if(!node.empty())
    {
        if(node.isInt())
            mDIFFlowVisEnable = (node.operator int() != 0);
        else if(node.isString())
            mDIFFlowVisEnable = (node.string() == "1" || node.string() == "true" || node.string() == "True");
    }
    node = fSettings["DIF.flow_vis_step"];
    if(!node.empty() && node.isInt())
        mDIFFlowVisStep = std::max(4, std::min(128, node.operator int()));
    node = fSettings["DIF.flow_vis_threshold"];
    if(!node.empty() && node.isReal())
        mDIFFlowVisThreshold = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.flow_vis_scale"];
    if(!node.empty() && node.isReal())
        mDIFFlowVisScale = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.flow_vis_max_len"];
    if(!node.empty() && node.isReal())
        mDIFFlowVisMaxLen = static_cast<float>(node.real());

    if(!mDIFSegCfg.enable)
        return true;

    node = fSettings["DIF.python"];
    if(!node.empty() && node.isString())
        mDIFSegCfg.python = node.string();

    node = fSettings["DIF.server_script"];
    if(!node.empty() && node.isString())
        mDIFSegCfg.server_script = node.string();

    node = fSettings["DIF.weights"];
    if(!node.empty() && node.isString())
        mDIFSegCfg.weights = node.string();

    node = fSettings["DIF.device"];
    if(!node.empty() && node.isString())
        mDIFSegCfg.device = node.string();

    node = fSettings["DIF.every_n_frames"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.every_n_frames = std::max(1, node.operator int());

    node = fSettings["DIF.jpeg_quality"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.jpeg_quality = std::max(1, std::min(100, node.operator int()));

    node = fSettings["DIF.req_timeout_ms"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.req_timeout_ms = std::max(100, node.operator int());

    node = fSettings["DIF.restart_backoff_ms"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.restart_backoff_ms = std::max(0, node.operator int());

    node = fSettings["DIF.imgsz"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.imgsz = std::max(64, node.operator int());

    node = fSettings["DIF.conf"];
    if(!node.empty() && node.isReal())
        mDIFSegCfg.conf = static_cast<float>(node.real());

    node = fSettings["DIF.iou"];
    if(!node.empty() && node.isReal())
        mDIFSegCfg.iou = static_cast<float>(node.real());

    node = fSettings["DIF.retina_masks"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.retina_masks = (node.operator int() != 0);

    node = fSettings["DIF.half"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.half = (node.operator int() != 0);

    node = fSettings["DIF.deterministic"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.deterministic = (node.operator int() != 0);

    node = fSettings["DIF.everything_raw_vis_enable"];
    if(!node.empty())
    {
        if(node.isInt())
            mDIFSegCfg.everything_raw_vis_enable = (node.operator int() != 0);
        else if(node.isString())
            mDIFSegCfg.everything_raw_vis_enable = (node.string() == "1" || node.string() == "true" || node.string() == "True");
    }

    node = fSettings["DIF.post_processing_vis_enable"];
    if(!node.empty())
    {
        if(node.isInt())
            mDIFSegCfg.post_processing_vis_enable = (node.operator int() != 0);
        else if(node.isString())
            mDIFSegCfg.post_processing_vis_enable = (node.string() == "1" || node.string() == "true" || node.string() == "True");
    }

    node = fSettings["DIF.binary_mask_vis_enable"];
    if(!node.empty())
    {
        if(node.isInt())
            mDIFBinaryMaskVisEnable = (node.operator int() != 0);
        else if(node.isString())
            mDIFBinaryMaskVisEnable = (node.string() == "1" || node.string() == "true" || node.string() == "True");
    }

    node = fSettings["DIF.area_min"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.area_min = std::max(1, node.operator int());

    node = fSettings["DIF.area_max_ratio"];
    if(!node.empty() && node.isReal())
        mDIFSegCfg.area_max_ratio = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.border_touch_min_sides"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.border_touch_min_sides = node.operator int();

    node = fSettings["DIF.morph_kernel"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.morph_kernel = std::max(0, node.operator int());

    node = fSettings["DIF.iou_nms"];
    if(!node.empty() && node.isReal())
        mDIFSegCfg.iou_nms = static_cast<float>(node.real());

    node = fSettings["DIF.max_masks"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.max_masks = std::max(1, node.operator int());

    node = fSettings["DIF.cc_min_area"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.cc_min_area = std::max(0, node.operator int());

    node = fSettings["DIF.min_assign_pixels"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.min_assign_pixels = std::max(0, node.operator int());

    node = fSettings["DIF.min_assign_ratio"];
    if(!node.empty() && node.isReal())
        mDIFSegCfg.min_assign_ratio = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.island_min_area"];
    if(!node.empty() && node.isInt())
        mDIFSegCfg.island_min_area = std::max(0, node.operator int());

    node = fSettings["DIF.debug_mask_mode"];
    if(!node.empty() && node.isInt())
        mDIFDebugMaskMode = std::max(0, node.operator int());

    node = fSettings["DIF.debug_dilate"];
    if(!node.empty() && node.isInt())
        mDIFDebugDilate = std::max(0, node.operator int());

    node = fSettings["DIF.filter_enable"];
    if(!node.empty() && node.isInt())
        mDIFFilterEnable = (node.operator int() != 0);

    node = fSettings["DIF.filter_max_lag"];
    if(!node.empty() && node.isInt())
        mDIFFilterMaxLag = std::max(0, node.operator int());

    node = fSettings["DIF.viewer_overlay"];
    if(!node.empty() && node.isInt())
        mDIFViewerOverlayMode = std::max(0, std::min(2, node.operator int()));

    node = fSettings["DIF.viewer_base_rgb"];
    if(!node.empty() && node.isInt())
        mDIFViewerBaseRGB = (node.operator int() != 0);

    node = fSettings["DIF.viewer_keypoints"];
    if(!node.empty() && node.isInt())
        mDIFViewerShowKeypoints = (node.operator int() != 0);

    node = fSettings["DIF.viewer_state_style"];
    if(!node.empty() && node.isInt())
        mDIFViewerStateStyle = std::max(0, std::min(1, node.operator int()));

    node = fSettings["DIF.viewer_inst_fill_enable"];
    if(!node.empty() && node.isInt())
        mDIFViewerInstFillEnable = (node.operator int() != 0);

    node = fSettings["DIF.viewer_inst_contour_enable"];
    if(!node.empty() && node.isInt())
        mDIFViewerInstContourEnable = (node.operator int() != 0);

    node = fSettings["DIF.viewer_inst_contour_lock_enable"];
    if(!node.empty() && node.isInt())
        mDIFViewerInstContourLockEnable = (node.operator int() != 0);

    node = fSettings["DIF.viewer_inst_label_enable"];
    if(!node.empty() && node.isInt())
        mDIFViewerInstLabelEnable = (node.operator int() != 0);

    node = fSettings["DIF.viewer_hud_enable"];
    if(!node.empty() && node.isInt())
        mDIFViewerHudEnable = (node.operator int() != 0);

    auto read_float01 = [&](const cv::FileNode& n, float& out)
    {
        if(n.empty())
            return;
        float v = out;
        if(n.isReal())
            v = static_cast<float>(n.real());
        else if(n.isInt())
            v = static_cast<float>(n.operator int());
        else
            return;
        out = std::max(0.0f, std::min(1.0f, v));
    };

    node = fSettings["DIF.viewer_alpha_S"];
    read_float01(node, mDIFViewerAlphaS);

    node = fSettings["DIF.viewer_alpha_lock"];
    read_float01(node, mDIFViewerAlphaLock);

    node = fSettings["DIF.viewer_alpha_D"];
    read_float01(node, mDIFViewerAlphaD);

    node = fSettings["DIF.viewer_label_min_area_ratio"];
    read_float01(node, mDIFViewerLabelMinAreaRatio);

    node = fSettings["DIF.viewer_label_min_area_ratio_S"];
    read_float01(node, mDIFViewerLabelMinAreaRatioS);

    auto parse_filtered_draw_mode = [&](const cv::FileNode& n)
    {
        if(n.empty())
            return;
        if(n.isInt())
        {
            mDIFViewerFilteredFeaturesMode = std::max(0, std::min(1, n.operator int()));
            return;
        }
        if(!n.isString())
            return;
        std::string mode = n.string();
        for(char& c : mode)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if(mode == "tracked_only" || mode == "tracked" || mode == "0")
            mDIFViewerFilteredFeaturesMode = 0;
        else if(mode == "static_candidates" || mode == "candidates" || mode == "1")
            mDIFViewerFilteredFeaturesMode = 1;
    };

    node = fSettings["DIF.viewer_filtered_draw_mode"];
    if(node.empty())
        node = fSettings["DIF.Viewer.FilteredDrawMode"];
    parse_filtered_draw_mode(node);

    node = fSettings["DIF.viewer_filtered_max_draw"];
    if(node.empty())
        node = fSettings["DIF.Viewer.MaxDrawFiltered"];
    if(!node.empty())
    {
        int max_draw = mDIFViewerFilteredFeaturesMaxDraw;
        if(node.isInt())
            max_draw = node.operator int();
        else if(node.isReal())
            max_draw = static_cast<int>(node.real());
        else if(node.isString())
        {
            const std::string s = node.string();
            char* end = nullptr;
            const long v = std::strtol(s.c_str(), &end, 10);
            if(end != s.c_str() && end && *end == '\0')
                max_draw = static_cast<int>(v);
        }
        mDIFViewerFilteredFeaturesMaxDraw = std::max(0, max_draw);
    }

    node = fSettings["DIF.sync"];
    if(!node.empty() && node.isInt())
        mDIFSync = (node.operator int() != 0);

    node = fSettings["DIF.sync_timeout_ms"];
    if(!node.empty() && node.isInt())
        mDIFSyncTimeoutMs = std::max(0, node.operator int());

    node = fSettings["DIF.initial_sync"];
    if(!node.empty() && node.isInt())
        mDIFInitialSync = (node.operator int() != 0);

    node = fSettings["DIF.initial_sync_timeout_ms"];
    if(!node.empty() && node.isInt())
        mDIFInitialSyncTimeoutMs = std::max(0, node.operator int());

    node = fSettings["DIF.second_pass_enable"];
    if(!node.empty() && node.isInt())
        mDIFSecondPassEnable = (node.operator int() != 0);

    node = fSettings["DIF.second_pass_enable_imu"];
    if(!node.empty() && node.isInt())
        mDIFSecondPassEnableIMU = (node.operator int() != 0);

    node = fSettings["DIF.second_pass_min_inliers"];
    if(!node.empty() && node.isInt())
        mDIFSecondPassMinInliers = std::max(0, node.operator int());

    node = fSettings["DIF.second_pass_iou_min"];
    if(!node.empty() && node.isReal())
        mDIFSecondPassIouMin = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.second_pass_dyn_ratio_delta"];
    if(!node.empty() && node.isReal())
        mDIFSecondPassDynRatioDelta = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.second_pass_max_lag"];
    if(!node.empty() && node.isInt())
        mDIFSecondPassMaxLag = std::max(0, node.operator int());

    // -----------------------
    // Module B: global tracks
    // -----------------------
    node = fSettings["DIF.B.enable"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.enable = (node.operator int() != 0);

    node = fSettings["DIF.B.area_min"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.area_min = std::max(1, node.operator int());

    node = fSettings["DIF.B.max_masks"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.max_masks = std::max(1, node.operator int());

    node = fSettings["DIF.B.n_feat_min"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.n_feat_min = std::max(0, node.operator int());

    node = fSettings["DIF.B.bbox_min_w"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.bbox_min_w = std::max(0, node.operator int());

    node = fSettings["DIF.B.bbox_min_h"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.bbox_min_h = std::max(0, node.operator int());

    node = fSettings["DIF.B.fill_ratio_min"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.fill_ratio_min = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.B.sample_stride"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.sample_stride = std::max(1, node.operator int());

    node = fSettings["DIF.B.sample_max_points"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.sample_max_points = std::max(0, node.operator int());

    node = fSettings["DIF.B.depth_min"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.depth_min = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["DIF.B.depth_max"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.depth_max = std::max(mDIFTrackCfg.depth_min, static_cast<float>(node.real()));

    node = fSettings["DIF.B.n_in_min"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.n_in_min = std::max(1, node.operator int());

    node = fSettings["DIF.B.mad_tau"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.mad_tau = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["DIF.B.w_iou"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.w_iou = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.w_2d"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.w_2d = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.w_3d"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.w_3d = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["DIF.B.sigma_2d"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.sigma_2d = std::max(1.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.sigma_3d"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.sigma_3d = std::max(1e-3f, static_cast<float>(node.real()));

    node = fSettings["DIF.B.tau_iou_min"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.tau_iou_min = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.tau_2d_max"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.tau_2d_max = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.gate_3d_max"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.gate_3d_max = static_cast<float>(node.real());
    node = fSettings["DIF.B.bbox_area_ratio_max"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.bbox_area_ratio_max = static_cast<float>(node.real());
    node = fSettings["DIF.B.v_max"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.v_max = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.margin_3d"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.margin_3d = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["DIF.B.cost_accept"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.cost_accept = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.B.max_miss"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.max_miss = std::max(1, node.operator int());

    node = fSettings["DIF.B.vel_ema_beta"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vel_ema_beta = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.B.vobs_iou_min"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_iou_min = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.B.vobs_mode"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.vobs_mode = std::max(0, std::min(2, node.operator int()));

    node = fSettings["DIF.B.vobs_px_sigma"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_px_sigma = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.vobs_dt_min"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_dt_min = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["DIF.B.vobs_match_n_min"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.vobs_match_n_min = std::max(1, node.operator int());
    node = fSettings["DIF.B.vobs_match_n_max"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.vobs_match_n_max = std::max(0, node.operator int());
    node = fSettings["DIF.B.vobs_depth_jump_ratio_max"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_depth_jump_ratio_max = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.vobs_mask_erode_px"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.vobs_mask_erode_px = std::max(0, node.operator int());
    // Alias (docs/参考资料/DIF_SLAM_工程实现方案.md v2.0): DIF.B.mask_erode_px -> DIF.B.vobs_mask_erode_px
    node = fSettings["DIF.B.mask_erode_px"];
    if(!node.empty())
    {
        int v = mDIFTrackCfg.vobs_mask_erode_px;
        if(node.isInt())
            v = node.operator int();
        else if(node.isReal())
            v = static_cast<int>(std::lround(node.real()));
        mDIFTrackCfg.vobs_mask_erode_px = std::max(0, v);
    }
    node = fSettings["DIF.B.vobs_edge_ratio_min"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_edge_ratio_min = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
    node = fSettings["DIF.B.vobs_mad_kappa"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_mad_kappa = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.vobs_trim_frac"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_trim_frac = std::max(0.0f, std::min(0.49f, static_cast<float>(node.real())));
    node = fSettings["DIF.B.vobs_depth_sigma_m"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_depth_sigma_m = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.vobs_sigma_r_max"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_sigma_r_max = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.vobs_inlier_ratio_min"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_inlier_ratio_min = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
    node = fSettings["DIF.B.vobs_q_min"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_q_min = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
    node = fSettings["DIF.B.vobs_q_n_ref"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.vobs_q_n_ref = std::max(1, node.operator int());
    node = fSettings["DIF.B.vobs_q_sigma_ref"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_q_sigma_ref = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.vobs_bi_label_enable"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.vobs_bi_label_enable = (node.operator int() != 0);

    node = fSettings["DIF.B.vobs_bg_suppress_enable"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.vobs_bg_suppress_enable = (node.operator int() != 0);
    node = fSettings["DIF.B.vobs_bg_area_ratio_min"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_bg_area_ratio_min = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
    node = fSettings["DIF.B.vobs_bg_z_sigma_max"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_bg_z_sigma_max = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.B.vobs_bg_v_min"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.vobs_bg_v_min = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["DIF.B.pose_inliers_min"];
    if(!node.empty() && node.isInt())
        mDIFTrackCfg.pose_inliers_min = std::max(0, node.operator int());

    node = fSettings["DIF.B.pose_rbg_max"];
    if(!node.empty() && node.isReal())
        mDIFTrackCfg.pose_rbg_max = static_cast<float>(node.real());

    // -----------------------
    // Module C: HMM + breaker
    // -----------------------
    node = fSettings["DIF.C.enable"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.enable = (node.operator int() != 0);

    node = fSettings["DIF.C.state_mode"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.state_mode = std::max(0, std::min(1, node.operator int()));

    // 2-state HMM transition matrix (state_mode=1)
    node = fSettings["DIF.C.A2_SS"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.A2[0] = std::max(1e-6f, std::min(1.0f, static_cast<float>(node.real())));
    node = fSettings["DIF.C.A2_SD"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.A2[1] = std::max(1e-6f, std::min(1.0f, static_cast<float>(node.real())));
    node = fSettings["DIF.C.A2_DS"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.A2[2] = std::max(1e-6f, std::min(1.0f, static_cast<float>(node.real())));
    node = fSettings["DIF.C.A2_DD"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.A2[3] = std::max(1e-6f, std::min(1.0f, static_cast<float>(node.real())));

    // Aliases (docs/参考资料/DIF_SLAM_工程实现方案.md v2.0): DIF.C.A_SS/A_SD/A_DS/A_DD -> A2
    if(mDIFStateCfg.state_mode == 1)
    {
        node = fSettings["DIF.C.A_SS"];
        if(!node.empty() && node.isReal())
            mDIFStateCfg.A2[0] = std::max(1e-6f, std::min(1.0f, static_cast<float>(node.real())));
        node = fSettings["DIF.C.A_SD"];
        if(!node.empty() && node.isReal())
            mDIFStateCfg.A2[1] = std::max(1e-6f, std::min(1.0f, static_cast<float>(node.real())));
        node = fSettings["DIF.C.A_DS"];
        if(!node.empty() && node.isReal())
            mDIFStateCfg.A2[2] = std::max(1e-6f, std::min(1.0f, static_cast<float>(node.real())));
        node = fSettings["DIF.C.A_DD"];
        if(!node.empty() && node.isReal())
            mDIFStateCfg.A2[3] = std::max(1e-6f, std::min(1.0f, static_cast<float>(node.real())));
    }

    // Map gating (state_mode=1): map_lock parameters
    node = fSettings["DIF.C.map_lock_enable"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.map_lock_enable = (node.operator int() != 0);
    node = fSettings["DIF.C.map_confirm_N"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.map_confirm_N = std::max(1, node.operator int());
    node = fSettings["DIF.C.map_cooldown_N"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.map_cooldown_N = std::max(0, node.operator int());

    node = fSettings["DIF.C.w_lock_min"];
    if(!node.empty())
    {
        float v = mDIFStateCfg.w_lock_min;
        if(node.isReal())
            v = static_cast<float>(node.real());
        else if(node.isInt())
            v = static_cast<float>(node.operator int());
        mDIFStateCfg.w_lock_min = std::max(0.0f, std::min(1.0f, v));
    }

    node = fSettings["DIF.C.dyn_mask_include_ever_dynamic"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.dyn_mask_include_ever_dynamic = (node.operator int() != 0);

    node = fSettings["DIF.C.suppress_ever_dynamic"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.suppress_ever_dynamic = (node.operator int() != 0);

    node = fSettings["DIF.C.suppress_enable"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.suppress_enable = (node.operator int() != 0);

    node = fSettings["DIF.C.prune_ever_dynamic"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.prune_ever_dynamic = (node.operator int() != 0);

    node = fSettings["DIF.C.sigma_S"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.sigma_S = std::max(1e-6f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.sigma_MS"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.sigma_MS = std::max(1e-6f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.mu_D"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.mu_D = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.sigma_D"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.sigma_D = std::max(1e-6f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.v_clip"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.v_clip = std::max(0.1f, static_cast<float>(node.real()));

    node = fSettings["DIF.C.v_bg_enable"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.v_bg_enable = (node.operator int() != 0);
    node = fSettings["DIF.C.v_bg_n_min"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.v_bg_n_min = std::max(1, node.operator int());
    node = fSettings["DIF.C.v_bg_trim_frac"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.v_bg_trim_frac = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
    // Alias (docs/参考资料/DIF_SLAM_工程实现方案.md v2.0): DIF.C.v_bg_frac -> DIF.C.v_bg_trim_frac
    node = fSettings["DIF.C.v_bg_frac"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.v_bg_trim_frac = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.C.tau_up"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.tau_up = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
    node = fSettings["DIF.C.tau_down"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.tau_down = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
    // Aliases (docs/参考资料/DIF_SLAM_工程实现方案.md v2.0): DIF.C.hyst_tau_up/down -> DIF.C.tau_up/down
    if(mDIFStateCfg.state_mode == 1)
    {
        node = fSettings["DIF.C.hyst_tau_up"];
        if(!node.empty() && node.isReal())
            mDIFStateCfg.tau_up = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
        node = fSettings["DIF.C.hyst_tau_down"];
        if(!node.empty() && node.isReal())
            mDIFStateCfg.tau_down = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
    }

    node = fSettings["DIF.C.mature_min"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.mature_min = std::max(0, node.operator int());

    node = fSettings["DIF.C.n_min"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.n_min = std::max(1, node.operator int());
    node = fSettings["DIF.C.n_bg_min"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.n_bg_min = std::max(1, node.operator int());

    node = fSettings["DIF.C.r_bg_fallback_enable"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.r_bg_fallback_enable = (node.operator int() != 0);
    node = fSettings["DIF.C.r_bg_trim_frac"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.r_bg_trim_frac = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.C.tau_bg"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.tau_bg = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.tau_q"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.tau_q = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.tau_dr"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.tau_dr = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.tau_r_m_min"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.tau_r_m_min = static_cast<float>(node.real());
    node = fSettings["DIF.C.K"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.K = std::max(1, node.operator int());
    node = fSettings["DIF.C.lambda_D"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.lambda_D = std::max(1e-6f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.lambda_other"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.lambda_other = std::max(1e-12f, static_cast<float>(node.real()));

    node = fSettings["DIF.C.flow_enable"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.flow_enable = (node.operator int() != 0);
    node = fSettings["DIF.C.flow_mode"];
    if(!node.empty())
    {
        if(node.isInt())
        {
            mDIFStateCfg.flow_mode = std::max(0, std::min(1, node.operator int()));
        }
        else if(node.isString())
        {
            std::string mode = node.string();
            for(char& c : mode)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if(mode == "desc" || mode == "orb" || mode == "match" || mode == "0")
                mDIFStateCfg.flow_mode = 0;
            else if(mode == "lk" || mode == "klt" || mode == "pyrlk" || mode == "1")
                mDIFStateCfg.flow_mode = 1;
        }
    }
    node = fSettings["DIF.C.flow_n_min"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.flow_n_min = std::max(1, node.operator int());
    node = fSettings["DIF.C.flow_n_bg_min"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.flow_n_bg_min = std::max(1, node.operator int());

    node = fSettings["DIF.C.flow_bg_fallback_enable"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.flow_bg_fallback_enable = (node.operator int() != 0);

    node = fSettings["DIF.C.flow_tau_bg"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.flow_tau_bg = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.flow_tau_q"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.flow_tau_q = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.flow_tau_de"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.flow_tau_de = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.flow_tau_e_m_min"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.flow_tau_e_m_min = static_cast<float>(node.real());
    node = fSettings["DIF.C.flow_K"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.flow_K = std::max(1, node.operator int());
    node = fSettings["DIF.C.flow_bg_trim_frac"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.flow_bg_trim_frac = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));

    node = fSettings["DIF.C.static_enable"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.static_enable = (node.operator int() != 0);
    node = fSettings["DIF.C.static_K"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.static_K = std::max(1, node.operator int());
    node = fSettings["DIF.C.static_tau_q"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.static_tau_q = std::max(1.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.static_tau_dr"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.static_tau_dr = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.static_flow_tau_q"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.static_flow_tau_q = std::max(1.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.static_flow_tau_de"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.static_flow_tau_de = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.static_lambda_S"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.static_lambda_S = std::max(1e-6f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.static_lambda_MS"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.static_lambda_MS = std::max(1e-6f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.static_lambda_D"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.static_lambda_D = std::max(1e-6f, static_cast<float>(node.real()));

    node = fSettings["DIF.C.static_hyst_tau_up"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.static_hyst_tau_up = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
    node = fSettings["DIF.C.static_hyst_tau_down"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.static_hyst_tau_down = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
    // Aliases (docs/参考资料/DIF_SLAM_工程实现方案.md v2.0): static_tau_up/down -> static_hyst_tau_up/down
    if(mDIFStateCfg.state_mode == 1)
    {
        node = fSettings["DIF.C.static_tau_up"];
        if(!node.empty() && node.isReal())
            mDIFStateCfg.static_hyst_tau_up = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
        node = fSettings["DIF.C.static_tau_down"];
        if(!node.empty() && node.isReal())
            mDIFStateCfg.static_hyst_tau_down = std::max(0.0f, std::min(1.0f, static_cast<float>(node.real())));
        node = fSettings["DIF.C.static_factor_S"];
        if(!node.empty() && node.isReal())
            mDIFStateCfg.static_lambda_S = std::max(0.0f, static_cast<float>(node.real()));
        node = fSettings["DIF.C.static_factor_D"];
        if(!node.empty() && node.isReal())
            mDIFStateCfg.static_lambda_D = std::max(0.0f, static_cast<float>(node.real()));
        node = fSettings["DIF.C.static_need_consecutive"];
        if(!node.empty() && node.isInt())
            mDIFStateCfg.static_K = std::max(1, node.operator int());
    }

    // Aliases (docs/参考资料/DIF_SLAM_工程实现方案.md v2.0): DIF.C.fuse.* -> internal breaker params
    node = fSettings["DIF.C.fuse.bg_r_max_px"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.tau_bg = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.fuse.q_r_min"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.tau_q = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.fuse.delta_r_min_px"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.tau_dr = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.fuse.bg_e_max_px"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.flow_tau_bg = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.fuse.q_e_min"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.flow_tau_q = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.fuse.delta_e_min_px"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.flow_tau_de = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.fuse.K_consecutive"];
    if(!node.empty() && node.isInt())
    {
        const int k = std::max(1, node.operator int());
        mDIFStateCfg.K = k;
        mDIFStateCfg.flow_K = k;
    }
    node = fSettings["DIF.C.fuse.lambda_D"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.lambda_D = std::max(0.0f, static_cast<float>(node.real()));
    node = fSettings["DIF.C.fuse.lambda_S"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.lambda_other = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["DIF.C.enter_D_K"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.enter_D_K = std::max(1, node.operator int());

    node = fSettings["DIF.C.mask_dilate"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.mask_dilate = std::max(0, node.operator int());

    node = fSettings["DIF.C.mask_dilate_per_lag"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.mask_dilate_per_lag = std::max(0, node.operator int());

    node = fSettings["DIF.C.mask_max_age_s"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.mask_max_age_s = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["DIF.C.mask_warp_enable"];
    if(!node.empty() && node.isInt())
        mDIFStateCfg.mask_warp_enable = (node.operator int() != 0);

    node = fSettings["DIF.C.mask_warp_max_dt"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.mask_warp_max_dt = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["DIF.C.mask_warp_max_px"];
    if(!node.empty() && node.isReal())
        mDIFStateCfg.mask_warp_max_px = std::max(0.0f, static_cast<float>(node.real()));

    // -----------------------
    // Dense mapping
    // -----------------------
    node = fSettings["dense.enable"];
    if(!node.empty())
    {
        if(node.isInt())
            mDenseCfg.enable = (node.operator int() != 0);
        else if(node.isString())
            mDenseCfg.enable = (node.string() == "1" || node.string() == "true" || node.string() == "True");
    }

    node = fSettings["dense.integrate_on"];
    if(!node.empty() && node.isString())
    {
        const std::string mode = node.string();
        if(mode == "keyframe" || mode == "segframe")
            mDenseCfg.integrate_on = mode;
    }

    node = fSettings["dense.voxel_size_m"];
    if(!node.empty() && node.isReal())
        mDenseCfg.voxel_size_m = std::max(1e-4f, static_cast<float>(node.real()));

    node = fSettings["dense.stride"];
    if(!node.empty() && node.isInt())
        mDenseCfg.stride = std::max(1, node.operator int());

    node = fSettings["dense.depth_min"];
    if(!node.empty() && node.isReal())
        mDenseCfg.depth_min = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["dense.depth_max"];
    if(!node.empty() && node.isReal())
        mDenseCfg.depth_max = std::max(mDenseCfg.depth_min, static_cast<float>(node.real()));

    node = fSettings["dense.forbid_dilate_px"];
    if(!node.empty() && node.isInt())
        mDenseCfg.forbid_dilate_px = std::max(0, node.operator int());

    node = fSettings["dense.forbid_fill_bbox"];
    if(!node.empty() && node.isInt())
        mDenseCfg.forbid_fill_bbox = (node.operator int() != 0);

    node = fSettings["dense.forbid_bbox_margin_px"];
    if(!node.empty() && node.isInt())
        mDenseCfg.forbid_bbox_margin_px = std::max(0, node.operator int());

    node = fSettings["dense.N_dense_confirm"];
    if(!node.empty() && node.isInt())
        mDenseCfg.n_dense_confirm = std::max(1, node.operator int());

    node = fSettings["dense.require_exact_label_map"];
    if(!node.empty() && node.isInt())
        mDenseCfg.require_exact_label_map = (node.operator int() != 0);

    node = fSettings["dense.queue_max"];
    if(!node.empty() && node.isInt())
        mDenseCfg.queue_max = std::max(0, node.operator int());

    node = fSettings["dense.rebuild_after_loop"];
    if(!node.empty() && node.isInt())
        mDenseCfg.rebuild_after_loop = (node.operator int() != 0);

    node = fSettings["dense.defusion_enable"];
    if(!node.empty() && node.isInt())
        mDenseCfg.defusion_enable = (node.operator int() != 0);

    node = fSettings["dense.instance_guard_px"];
    if(!node.empty() && node.isInt())
        mDenseCfg.instance_guard_px = std::max(0, node.operator int());

    node = fSettings["dense.min_voxel_weight"];
    if(!node.empty() && node.isInt())
        mDenseCfg.min_voxel_weight = std::max(1, node.operator int());

    node = fSettings["dense.min_voxel_frames"];
    if(!node.empty() && node.isInt())
        mDenseCfg.min_voxel_frames = std::max(0, node.operator int());

    node = fSettings["dense.depth_median_ksize"];
    if(!node.empty() && node.isInt())
        mDenseCfg.depth_median_ksize = std::max(0, node.operator int());

    node = fSettings["dense.depth_edge_delta_m"];
    if(!node.empty() && node.isReal())
        mDenseCfg.depth_edge_delta_m = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["dense.depth_edge_delta_ratio"];
    if(!node.empty() && node.isReal())
        mDenseCfg.depth_edge_delta_ratio = std::max(0.0f, static_cast<float>(node.real()));

    node = fSettings["dense.depth_edge_min_neighbors"];
    if(!node.empty() && node.isInt())
        mDenseCfg.depth_edge_min_neighbors = std::max(0, node.operator int());

    node = fSettings["dense.min_voxel_neighbors"];
    if(!node.empty() && node.isInt())
        mDenseCfg.min_voxel_neighbors = std::max(0, node.operator int());

    return true;
}

int Tracking::CountDIFFrameInliers(const Frame& frame) const
{
    int n = 0;
    const int N = frame.N;
    for(int i = 0; i < N; ++i)
    {
        MapPoint* mp = (i >= 0 && i < static_cast<int>(frame.mvpMapPoints.size())) ? frame.mvpMapPoints[i] : nullptr;
        if(!mp)
            continue;
        if(i >= 0 && i < static_cast<int>(frame.mvbOutlier.size()) && frame.mvbOutlier[i])
            continue;

        if(mbOnlyTracking)
        {
            n++;
            continue;
        }

        if(mp->Observations() > 0)
            n++;
    }
    return n;
}

void Tracking::MaybeEnqueueDenseMappingFromKeyFrame(KeyFrame* pKF)
{
    if(!mpDenseMapping || !mDenseCfg.enable)
        return;
    if(mDenseCfg.integrate_on != "keyframe")
        return;
    if(!pKF)
        return;
    if(mImRGB.empty() || mImDepth.empty())
        return;
    if(!mCurrentFrame.HasPose())
        return;

    const int fid = static_cast<int>(pKF->mnFrameId);
    DenseKFPacket packet;
    if(BuildDensePacketFromFrame(fid,
                                  pKF->mTimeStamp,
                                  mImRGB,
                                  mImDepth,
                                  pKF->GetPose(),
                                  pKF,
                                  packet))
    {
        mpDenseMapping->Enqueue(std::move(packet));
        mDensePendingKeyFrames.erase(fid);
        return;
    }

    // DIF-SLAM v2.0 exact-label mode requires each integrated KeyFrame to have its own aligned
    // label_map/local_to_global. In async/low-rate mode, avoid forcing a segmentation here unless exact
    // alignment was explicitly requested; otherwise this quietly turns keyframe insertion into a blocking FastSAM call.
    if(mDenseCfg.require_exact_label_map && mDIFSegCfg.enable && mpDIFSegWorker && !HasDIFSegMappingForFrameId(fid))
    {
        SegmentationResult seg;
        mpDIFSegWorker->SubmitFrame(fid, pKF->mTimeStamp, mImRGB, mbRGB);
        const int timeout_ms = (mDIFSyncTimeoutMs > 0) ? mDIFSyncTimeoutMs : 8000;
        if(mpDIFSegWorker->WaitForFrameResult(fid, timeout_ms, seg) && seg.ok && !seg.label_map.empty())
        {
            StoreDIFPendingSegmentation(seg);
            ProcessDIFPendingSegmentations(); // will populate seg-cache + local_to_global

            DenseKFPacket packet2;
            if(BuildDensePacketFromFrame(fid,
                                         pKF->mTimeStamp,
                                         mImRGB,
                                         mImDepth,
                                         pKF->GetPose(),
                                         pKF,
                                         packet2))
            {
                mpDenseMapping->Enqueue(std::move(packet2));
                mDensePendingKeyFrames.erase(fid);
                return;
            }
        }
    }

    // Fallback: defer until ProcessDIFPendingSegmentations stores the mapping (best-effort, may be dropped if stale).
    DensePendingKeyFrame pending;
    pending.pKF = pKF;
    pending.frame_id = fid;
    pending.timestamp = pKF->mTimeStamp;
    pending.rgb = mImRGB;
    pending.depth = mImDepth;
    mDensePendingKeyFrames[fid] = std::move(pending);
}

void Tracking::MaybeFlushPendingDenseKeyFrame(int frame_id)
{
    if(!mpDenseMapping || !mDenseCfg.enable)
        return;
    if(mDenseCfg.integrate_on != "keyframe")
        return;
    if(mDensePendingKeyFrames.empty())
        return;

    // A new segmentation mapping for `frame_id` has just been stored. Since `dense.require_exact_label_map`
    // may allow +/-1 frame alignment, try flushing all pending keyframes opportunistically.
    (void)frame_id;
    for(auto it = mDensePendingKeyFrames.begin(); it != mDensePendingKeyFrames.end(); )
    {
        const DensePendingKeyFrame& pending = it->second;
        if(!pending.pKF || pending.rgb.empty() || pending.depth.empty())
        {
            it = mDensePendingKeyFrames.erase(it);
            continue;
        }

        DenseKFPacket packet;
        if(!BuildDensePacketFromFrame(pending.frame_id,
                                      pending.timestamp,
                                      pending.rgb,
                                      pending.depth,
                                      pending.pKF->GetPose(),
                                      pending.pKF,
                                      packet))
        {
            ++it;
            continue;
        }

        mpDenseMapping->Enqueue(std::move(packet));
        it = mDensePendingKeyFrames.erase(it);
    }
}

void Tracking::MaybeEnqueueDenseMappingFromSegFrame(int frame_id,
                                                    double timestamp,
                                                    const cv::Mat& label_map,
                                                    const Tracking::DIFFrameCacheEntry& fe,
                                                    const std::vector<int>& local_to_global)
{
    if(!mpDenseMapping || !mDenseCfg.enable)
        return;
    if(mDenseCfg.integrate_on != "segframe")
        return;
    if(label_map.empty())
        return;
    if(!fe.has_pose || fe.depth.empty())
        return;
    if(fe.rgb.empty())
        return;
    if(local_to_global.empty())
        return;
    if(!mpDIFInstanceTracker || !mDIFStateCfg.enable)
        return;

    DenseKFPacket packet;
    packet.kf_id = -1;
    packet.frame_id = frame_id;
    packet.timestamp = timestamp;
    packet.has_pose = true;
    packet.Tcw = fe.Tcw;
    packet.rgb = fe.rgb;
    packet.depth = fe.depth;
    packet.label_map = label_map;
    packet.local_to_global = local_to_global;
    packet.forbid_mask = BuildDIFForbidMaskForDense(label_map.rows, label_map.cols, timestamp, fe.has_pose, fe.Tcw);
    packet.fx = Frame::fx;
    packet.fy = Frame::fy;
    packet.cx = Frame::cx;
    packet.cy = Frame::cy;
    packet.is_rgb = mbRGB;

    const auto& tracks = mpDIFInstanceTracker->GetTracks();
    packet.track_states.reserve(tracks.size());
    for(const auto& kv : tracks)
    {
        DenseTrackStateInfo info;
        info.state = kv.second.state_hat;
        info.map_lock = kv.second.map_lock;
        info.w_lock = kv.second.w_lock;
        packet.track_states.emplace(kv.first, info);
    }

    mpDenseMapping->Enqueue(std::move(packet));
}

bool Tracking::GetDenseSegmentationForFrame(int frame_id,
                                            cv::Mat& out_label_map,
                                            std::vector<int>& out_local_to_global,
                                            int& out_seg_frame_id) const
{
    out_seg_frame_id = -1;
    if(!mDIFSegCfg.enable)
        return false;

    const auto pick_frame = [&](int fid) -> bool {
        const auto it = mDIFSegCacheByFrameId.find(fid);
        if(it == mDIFSegCacheByFrameId.end())
            return false;
        if(it->second.label_map.empty() || it->second.local_to_global.empty())
            return false;
        if(it->second.label_map.type() != CV_16S)
            return false;
        out_label_map = it->second.label_map;
        out_local_to_global = it->second.local_to_global;
        out_seg_frame_id = fid;
        return true;
    };

    std::unique_lock<std::mutex> lock(mMutexDIFSegCache);
    if(pick_frame(frame_id))
        return true;
    if(mDenseCfg.require_exact_label_map)
        return false;
    if(pick_frame(frame_id - 1))
        return true;
    if(pick_frame(frame_id + 1))
        return true;
    return false;
}

bool Tracking::BuildDensePacketFromFrame(int frame_id,
                                         double timestamp,
                                         const cv::Mat& rgb,
                                         const cv::Mat& depth,
                                         const Sophus::SE3f& Tcw,
                                         KeyFrame* pKF,
                                         DenseKFPacket& out_packet)
{
    if(!mpDenseMapping || !mDenseCfg.enable)
        return false;
    if(!mpDIFInstanceTracker || !mDIFStateCfg.enable)
        return false;
    if(frame_id < 0 || rgb.empty() || depth.empty())
        return false;

    cv::Mat label_map;
    std::vector<int> local_to_global;
    int seg_frame_id = -1;
    if(!GetDenseSegmentationForFrame(frame_id, label_map, local_to_global, seg_frame_id))
        return false;

    out_packet.kf_id = pKF ? static_cast<int>(pKF->mnId) : -1;
    out_packet.frame_id = frame_id;
    out_packet.timestamp = timestamp;
    out_packet.has_pose = true;
    out_packet.Tcw = Tcw;
    out_packet.rgb = rgb;
    out_packet.depth = depth;
    out_packet.label_map = label_map;
    out_packet.local_to_global = local_to_global;
    out_packet.forbid_mask = BuildDIFForbidMaskForDense(label_map.rows, label_map.cols, timestamp, true, Tcw);
    out_packet.fx = Frame::fx;
    out_packet.fy = Frame::fy;
    out_packet.cx = Frame::cx;
    out_packet.cy = Frame::cy;
    out_packet.is_rgb = mbRGB;
    out_packet.pKF = pKF;

    const auto& tracks = mpDIFInstanceTracker->GetTracks();
    out_packet.track_states.reserve(tracks.size());
    for(const auto& kv : tracks)
    {
        DenseTrackStateInfo info;
        info.state = kv.second.state_hat;
        info.map_lock = kv.second.map_lock;
        info.w_lock = kv.second.w_lock;
        out_packet.track_states.emplace(kv.first, info);
    }

    return true;
}

void Tracking::MaybeRebuildDenseMappingAfterLoop()
{
    if(!mpDenseMapping || !mDenseCfg.enable || !mDenseCfg.rebuild_after_loop)
        return;
    if(!mpAtlas)
        return;
    const int idx = mpAtlas->GetLastBigChangeIdx();
    if(mDenseLastMapChangeIdx < 0)
    {
        mDenseLastMapChangeIdx = idx;
        return;
    }
    if(idx > mDenseLastMapChangeIdx)
    {
        mDenseLastMapChangeIdx = idx;
        mpDenseMapping->RebuildGlobalVoxels(true);
    }
}

void Tracking::SaveDIFBinaryMaskForCurrentFrame()
{
    if(!mDIFSegCfg.enable || !mDIFBinaryMaskVisEnable)
        return;

    const int fid = static_cast<int>(mCurrentFrame.mnId);
    if(fid < 0)
        return;

    const int height = mImGray.rows;
    const int width = mImGray.cols;
    if(height <= 0 || width <= 0)
        return;

    cv::Mat mask;
    if(mDIFStateCfg.enable && mpDIFInstanceTracker)
    {
        // Use the per-frame predicted M_t^{dyn} (D pixels only) built from current track states.
        // Semantics required by this project:
        // - 255 (white): target / pixels to be filtered out
        // - 0 (black): background / pixels to keep
        mask = GetDIFDynamicMaskPredForCurrentFrame();
    }
    else if(!mDIFDynMaskLast.empty())
    {
        // Fallback (e.g., debug mask path when Module C is disabled).
        mask = mDIFDynMaskLast;
    }

    if(mask.empty())
    {
        mask = cv::Mat(height, width, CV_8U, cv::Scalar(0));
    }
    else
    {
        if(mask.type() != CV_8U)
        {
            cv::Mat tmp;
            mask.convertTo(tmp, CV_8U);
            mask = tmp;
        }
        if(mask.rows != height || mask.cols != width)
        {
            cv::Mat tmp;
            cv::resize(mask, tmp, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
            mask = tmp;
        }
        cv::Mat bin = (mask != 0);
        bin.convertTo(mask, CV_8U, 255);
    }

    const std::string dir = "Output/binary_mask";
    if(!EnsureDir("Output") || !EnsureDir(dir))
    {
        static bool warned = false;
        if(!warned)
        {
            std::cerr << "[DIF] failed to create binary mask dir: " << dir << std::endl;
            warned = true;
        }
        return;
    }

    const std::string path = BuildFramePath(dir, fid);
    const std::string tmp_path = BuildTmpFramePath(dir, fid);

    if(!cv::imwrite(tmp_path, mask))
    {
        static int last_warn_fid = -1;
        if(last_warn_fid != fid)
        {
            std::cerr << "[DIF] failed to write binary mask: " << tmp_path << std::endl;
            last_warn_fid = fid;
        }
        return;
    }
    if(std::rename(tmp_path.c_str(), path.c_str()) != 0)
    {
        static int last_warn_fid = -1;
        if(last_warn_fid != fid)
        {
            std::cerr << "[DIF] failed to rename binary mask tmp file: " << tmp_path
                      << " -> " << path << std::endl;
            last_warn_fid = fid;
        }
        std::remove(tmp_path.c_str());
        return;
    }
}

void Tracking::SaveFlowVisForCurrentFrame()
{
    if(!mDIFFlowVisEnable)
        return;

    const int fid = static_cast<int>(mCurrentFrame.mnId);
    if(fid < 0)
        return;

    if(mImGray.empty())
        return;

    cv::Mat base_bgr;
    if(!mImRGB.empty())
    {
        if(mImRGB.type() == CV_8UC3)
            base_bgr = mImRGB;
        else if(mImRGB.channels() == 3)
            mImRGB.convertTo(base_bgr, CV_8UC3);
    }
    if(base_bgr.empty())
    {
        if(mImGray.type() == CV_8U)
            cv::cvtColor(mImGray, base_bgr, cv::COLOR_GRAY2BGR);
        else
        {
            cv::Mat gray8;
            mImGray.convertTo(gray8, CV_8U);
            cv::cvtColor(gray8, base_bgr, cv::COLOR_GRAY2BGR);
        }
    }

    cv::Mat vis = base_bgr.clone();

    // Only draw flow vectors inside the current dynamic mask (D-only).
    cv::Mat dyn_mask;
    if(mDIFSegCfg.enable && mDIFStateCfg.enable && mpDIFInstanceTracker)
    {
        const cv::Mat& m = GetDIFDynamicMaskPredForCurrentFrame(); // CV_8U, 0/255
        if(!m.empty())
            dyn_mask = m;
    }
    if(!dyn_mask.empty())
    {
        if(dyn_mask.type() != CV_8U)
        {
            cv::Mat tmp;
            dyn_mask.convertTo(tmp, CV_8U);
            dyn_mask = tmp;
        }
        if(dyn_mask.rows != vis.rows || dyn_mask.cols != vis.cols)
        {
            cv::Mat tmp;
            cv::resize(dyn_mask, tmp, vis.size(), 0, 0, cv::INTER_NEAREST);
            dyn_mask = tmp;
        }
    }

    const bool can_flow = !mImGrayPrev.empty() && mImGrayPrev.type() == CV_8U && mImGray.type() == CV_8U &&
                          mImGrayPrev.size() == mImGray.size();
    if(can_flow && !dyn_mask.empty())
    {
        const int step = std::max(4, std::min(128, mDIFFlowVisStep));
        const int half = step / 2;

        std::vector<cv::Point2f> p0;
        p0.reserve(static_cast<size_t>((mImGray.rows / step + 1) * (mImGray.cols / step + 1)));
        for(int y = half; y < mImGray.rows; y += step)
        {
            for(int x = half; x < mImGray.cols; x += step)
            {
                p0.emplace_back(static_cast<float>(x), static_cast<float>(y));
            }
        }

        std::vector<cv::Point2f> p1;
        std::vector<uchar> status;
        std::vector<float> err;

        const cv::Size win_size(21, 21);
        const int max_level = 3;
        const cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(mImGrayPrev, mImGray, p0, p1, status, err, win_size, max_level, termcrit, 0, 1e-4);

        const float thr = std::max(0.0f, mDIFFlowVisThreshold);
        const float scale = std::max(0.0f, mDIFFlowVisScale);
        const float max_len = mDIFFlowVisMaxLen;
        const cv::Scalar green(0, 255, 0);

        const int w = vis.cols;
        const int h = vis.rows;
        const size_t n = std::min(p0.size(), std::min(p1.size(), status.size()));
        for(size_t i = 0; i < n; ++i)
        {
            if(status[i] == 0)
                continue;
            const cv::Point2f a = p0[i];
            const cv::Point2f b = p1[i];
            const float dx0 = b.x - a.x;
            const float dy0 = b.y - a.y;
            const float mag0 = std::sqrt(dx0 * dx0 + dy0 * dy0);
            if(!(mag0 > thr) || !std::isfinite(mag0))
                continue;

            const int bx = static_cast<int>(b.x + 0.5f);
            const int by = static_cast<int>(b.y + 0.5f);
            if(bx < 0 || by < 0 || bx >= w || by >= h)
                continue;
            if(dyn_mask.at<uchar>(by, bx) == 0)
                continue;

            float dx = dx0 * scale;
            float dy = dy0 * scale;
            float mag = mag0 * scale;
            if(std::isfinite(max_len) && max_len > 0.0f && mag > max_len && mag > 1e-6f)
            {
                const float s = max_len / mag;
                dx *= s;
                dy *= s;
            }

            // Anchor arrows at the current point (b) so they stay within the dynamic region on the current frame.
            cv::Point2f end(b.x + dx, b.y + dy);
            end.x = std::max(0.0f, std::min(static_cast<float>(w - 1), end.x));
            end.y = std::max(0.0f, std::min(static_cast<float>(h - 1), end.y));

            const cv::Point pa(bx, by);
            const cv::Point pb(static_cast<int>(end.x + 0.5f), static_cast<int>(end.y + 0.5f));
            cv::arrowedLine(vis, pa, pb, green, 1, cv::LINE_AA, 0, 0.25);
        }
    }

    const std::string dir = "Output/flow_vis";
    if(!EnsureDir("Output") || !EnsureDir(dir))
    {
        static bool warned = false;
        if(!warned)
        {
            std::cerr << "[FLOW] failed to create flow_vis dir: " << dir << std::endl;
            warned = true;
        }
        return;
    }

    const std::string path = BuildFramePath(dir, fid);
    const std::string tmp_path = BuildTmpFramePath(dir, fid);
    if(!cv::imwrite(tmp_path, vis))
    {
        static int last_warn_fid = -1;
        if(last_warn_fid != fid)
        {
            std::cerr << "[FLOW] failed to write flow visualization: " << tmp_path << std::endl;
            last_warn_fid = fid;
        }
        return;
    }
    if(std::rename(tmp_path.c_str(), path.c_str()) != 0)
    {
        static int last_warn_fid = -1;
        if(last_warn_fid != fid)
        {
            std::cerr << "[FLOW] failed to rename flow vis tmp file: " << tmp_path << " -> " << path << std::endl;
            last_warn_fid = fid;
        }
        std::remove(tmp_path.c_str());
        return;
    }
}

void Tracking::MaybeDIFSecondPassPoseOptimization()
{
    if(!mDIFSegCfg.enable || !mDIFFilterEnable)
        return;
    if(!mDIFSecondPassEnable)
        return;
    if(!mDIFStateCfg.enable || !mpDIFStateEstimator || !mpDIFInstanceTracker)
        return;
    if(mDIFDynMaskLast.empty())
        return;
    const bool imu_initialized = (mpAtlas && mpAtlas->isImuInitialized());
    if(imu_initialized && !mDIFSecondPassEnableIMU)
        return;

    const int fid = static_cast<int>(mCurrentFrame.mnId);
    const int lag = (mDIFPrevDynMaskFrameId >= 0) ? (fid - mDIFPrevDynMaskFrameId) : -1;
    if(mDIFPrevDynMask.empty() || lag <= 0)
        return;
    if(mDIFSecondPassMaxLag > 0 && lag > mDIFSecondPassMaxLag)
        return;

    cv::Mat cur_mask = mDIFDynMaskLast;
    cv::Mat prev_mask = mDIFPrevDynMask;
    if(cur_mask.type() != CV_8U)
        cur_mask.convertTo(cur_mask, CV_8U);
    if(prev_mask.type() != CV_8U)
        prev_mask.convertTo(prev_mask, CV_8U);
    if(prev_mask.rows != cur_mask.rows || prev_mask.cols != cur_mask.cols)
        cv::resize(prev_mask, prev_mask, cur_mask.size(), 0, 0, cv::INTER_NEAREST);

    const int area_cur = cv::countNonZero(cur_mask);
    const int area_prev = cv::countNonZero(prev_mask);
    const int total_area = std::max(1, cur_mask.rows * cur_mask.cols);
    const float ratio_cur = static_cast<float>(area_cur) / static_cast<float>(total_area);
    const float ratio_prev = static_cast<float>(area_prev) / static_cast<float>(total_area);
    const float ratio_delta = std::abs(ratio_cur - ratio_prev);

    cv::Mat inter;
    cv::bitwise_and(cur_mask, prev_mask, inter);
    const int area_inter = cv::countNonZero(inter);
    const int area_union = area_cur + area_prev - area_inter;
    const float iou = (area_union > 0) ? (static_cast<float>(area_inter) / static_cast<float>(area_union)) : 1.0f;

    if(iou >= mDIFSecondPassIouMin && ratio_delta <= mDIFSecondPassDynRatioDelta)
        return;

    const int inliers_before = mnMatchesInliers;
    if(mDIFSecondPassMinInliers > 0 && inliers_before < mDIFSecondPassMinInliers)
        return;

    int before_nonnull = 0;
    for(MapPoint* mp : mCurrentFrame.mvpMapPoints)
        before_nonnull += (mp != nullptr);

    ApplyDIFMaskToMatchedMapPoints(mCurrentFrame); // uses current M_t^{dyn} (already updated in this frame)

    int after_nonnull = 0;
    for(MapPoint* mp : mCurrentFrame.mvpMapPoints)
        after_nonnull += (mp != nullptr);
    const int removed_extra = std::max(0, before_nonnull - after_nonnull);
    if(removed_extra <= 0)
        return;

    const char* opt_mode = "vis";
    if(!imu_initialized)
    {
        Optimizer::PoseOptimization(&mCurrentFrame);
    }
    else
    {
        if(mCurrentFrame.mnId <= mnLastRelocFrameId + mnFramesToResetIMU)
        {
            opt_mode = "vis_imu_reset";
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else if(!mbMapUpdated)
        {
            opt_mode = "imu_last_frame";
            (void)Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame);
        }
        else
        {
            opt_mode = "imu_last_kf";
            (void)Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame);
        }
    }
    if(mCurrentFrame.HasPose())
        mCurrentFrame.UpdatePoseMatrices();

    const int inliers_after = CountDIFFrameInliers(mCurrentFrame);
    mnMatchesInliers = inliers_after;
    if(mpLocalMapper)
        mpLocalMapper->mnMatchesInliers = mnMatchesInliers;

    // Update pose cache entry for potential late-arriving segmentation alignment.
    for(auto& e : mDIFFrameCache)
    {
        if(e.frame_id == fid)
        {
            e.has_pose = mCurrentFrame.HasPose();
            if(e.has_pose)
                e.Tcw = mCurrentFrame.GetPose();
            e.inliers = (mState == OK) ? mnMatchesInliers : 0;
            break;
        }
    }

    std::cout << "[DIF-2PASS] frame_id=" << fid
              << " imu=" << (imu_initialized ? 1 : 0)
              << " opt=" << opt_mode
              << " lag=" << lag
              << " iou=" << iou
              << " dyn_ratio_delta=" << ratio_delta
              << " removed_extra=" << removed_extra
              << " inliers_before=" << inliers_before
              << " inliers_after=" << inliers_after
              << std::endl;
}

void Tracking::UpdateDIFSegmentation()
{
    if(!mpDIFSegWorker)
        return;

    SegmentationResult seg;
    if(!mpDIFSegWorker->TryGetLatest(seg))
        return;

    const int cur_fid = static_cast<int>(mCurrentFrame.mnId);
    const int lag = (seg.frame_id >= 0) ? (cur_fid - seg.frame_id) : -1;

    if(seg.frame_id != mDIFLastPrintSegFrameId)
    {
        mDIFLastPrintSegFrameId = seg.frame_id;
        // Frame separator for better log readability
        std::cout << "------------------------ [DIF Frame " << seg.frame_id << "] ------------------------" << std::endl;
        if(seg.ok && !seg.label_map.empty())
        {
            std::cout << "[DIF] seg ok frame_id=" << seg.frame_id
                      << " cur=" << cur_fid
                      << " lag=" << lag
                      << " elapsed_ms=" << seg.elapsed_ms
                      << " infer_ms=" << seg.profile_infer_ms
                      << " post_ms=" << seg.profile_post_ms
                      << " label=" << seg.label_map.cols << "x" << seg.label_map.rows
                      << " debug_mask_mode=" << mDIFDebugMaskMode
                      << " filter_enable=" << (mDIFFilterEnable ? 1 : 0)
                      << std::endl;
        }
        else
        {
            std::cout << "[DIF] seg failed frame_id=" << seg.frame_id
                      << " cur=" << cur_fid
                      << " lag=" << lag
                      << " elapsed_ms=" << seg.elapsed_ms
                      << " error=" << seg.error
                      << std::endl;
        }
    }

    if(!seg.ok || seg.label_map.empty())
        return;

    UpdateDIFDebugMaskFromLabelMap(seg);
}

bool Tracking::IsInDIFMask(const cv::Mat& mask, const cv::Point2f& pt) const
{
    if(mask.empty() || mask.type() != CV_8U)
        return false;
    const int x = static_cast<int>(pt.x + 0.5f);
    const int y = static_cast<int>(pt.y + 0.5f);
    if(x < 0 || y < 0 || x >= mask.cols || y >= mask.rows)
        return false;
    return mask.at<uint8_t>(y, x) != 0;
}

void Tracking::UpdateDIFDebugMaskFromLabelMap(const SegmentationResult& seg)
{
    if(!seg.ok || seg.label_map.empty())
        return;

    cv::Mat label_raw = seg.label_map;
    if(label_raw.type() != CV_16S)
        return;

    StoreDIFPendingSegmentation(seg);
    StoreDIFSegCacheLabelMap(seg.frame_id, label_raw);

    cv::Mat label_vis = label_raw;
    if(!mImGray.empty() && (label_vis.rows != mImGray.rows || label_vis.cols != mImGray.cols))
    {
        cv::Mat resized;
        cv::resize(label_vis, resized, mImGray.size(), 0, 0, cv::INTER_NEAREST);
        label_vis = resized;
    }

    // Cache latest label_map for visualization (even if debug mask is off).
    mDIFLabelMapLast = label_vis;
    mDIFLastSegFrameId = seg.frame_id;
    mDIFLastSegTimestamp = seg.timestamp;
    mDIFLastSegElapsedMs = seg.elapsed_ms;

    // If module C is enabled, dynamic mask is produced after pose optimization from tracks.
    if(mDIFStateCfg.enable)
        return;

    if(mDIFDebugMaskMode <= 0)
    {
        mDIFDynMaskLast.release();
        mDIFLastDynMaskFrameId = -1;
        mDIFLastDynMaskTimestamp = -1.0;
        return;
    }

    cv::Mat dyn_mask(label_vis.rows, label_vis.cols, CV_8U, cv::Scalar(0));

    if(mDIFDebugMaskMode == 1)
    {
        dyn_mask = (label_vis >= 0);
        dyn_mask.convertTo(dyn_mask, CV_8U, 255);
    }
    else if(mDIFDebugMaskMode == 2)
    {
        int max_id = -1;
        int max_area = 0;
        const int total = label_vis.rows * label_vis.cols;
        const int16_t* p = label_vis.ptr<int16_t>(0);
        int cur_max = -1;
        int cur_min = -1;
        for(int i = 0; i < total; ++i)
        {
            const int v = static_cast<int>(p[i]);
            if(v < 0) continue;
            if(cur_min < 0 || v < cur_min) cur_min = v;
            if(cur_max < 0 || v > cur_max) cur_max = v;
        }
        if(cur_max >= 0)
        {
            std::vector<int> hist(static_cast<size_t>(cur_max + 1), 0);
            for(int i = 0; i < total; ++i)
            {
                const int v = static_cast<int>(p[i]);
                if(v >= 0 && v <= cur_max)
                    hist[static_cast<size_t>(v)]++;
            }
            for(int id = 0; id <= cur_max; ++id)
            {
                if(hist[static_cast<size_t>(id)] > max_area)
                {
                    max_area = hist[static_cast<size_t>(id)];
                    max_id = id;
                }
            }
        }

        if(max_id >= 0)
        {
            dyn_mask = (label_vis == max_id);
            dyn_mask.convertTo(dyn_mask, CV_8U, 255);
        }
    }

    if(mDIFDebugDilate > 0 && !dyn_mask.empty())
    {
        const int k = std::max(1, mDIFDebugDilate);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        cv::dilate(dyn_mask, dyn_mask, kernel);
    }

    mDIFDynMaskLast = dyn_mask;
    mDIFLastDynMaskFrameId = seg.frame_id;
    mDIFLastDynMaskTimestamp = seg.timestamp;
}

void Tracking::ApplyDIFMaskToMatchedMapPoints(Frame& frame)
{
    // 1) 让 filter_enable 真正生效（非常关键：否则无法做消融）
    if(!mDIFFilterEnable)
        return;

    const cv::Mat* dyn_mask = nullptr;
    const bool is_current_frame = (frame.mnId == mCurrentFrame.mnId);
    if(is_current_frame && mDIFStateCfg.enable && mpDIFInstanceTracker)
    {
        // Build a per-frame predicted M_t^{dyn} from tracks' last masks (with optional warp/age gating).
        // This mitigates segmentation lag/missing and avoids relying on a stale cached mask (P-002).
        dyn_mask = &GetDIFDynamicMaskPredForCurrentFrame();
    }
    else if(!mDIFDynMaskLast.empty())
    {
        // Fallback path: use the last stored mask (e.g., debug mode / when Module C is disabled).
        if(mDIFFilterMaxLag > 0 && mDIFLastDynMaskFrameId >= 0)
        {
            const int lag = static_cast<int>(frame.mnId) - mDIFLastDynMaskFrameId;
            if(lag > mDIFFilterMaxLag)
                return;
        }
        dyn_mask = &mDIFDynMaskLast;
    }
    if(!dyn_mask || dyn_mask->empty())
        return;

    if(frame.mvKeys.empty() || frame.mvpMapPoints.empty())
        return;

    // 2) 读取 suppressed 集合（按轨迹状态抑制）
    std::unique_lock<std::mutex> lock_suppress(mMutexDIFSuppressed);
    const std::unordered_set<int>& suppressed = mDIFSuppressedTrackIds;
    const bool use_state_suppress =
        (mDIFStateCfg.enable && mDIFStateCfg.suppress_enable && !suppressed.empty());

    // 3) 先收集"将要删除"的索引，再决定是否执行 mask 删除（护栏）
    const int N = frame.N;
    int n_before = 0;
    std::vector<int> idx_state;
    std::vector<int> idx_mask;
    idx_state.reserve(64);
    idx_mask.reserve(128);

    for(int i = 0; i < N; ++i)
    {
        MapPoint* pMP = frame.mvpMapPoints[i];
        if(!pMP)
            continue;
        // Count only effective constraints (ignore pre-flagged outliers).
        if(i >= 0 && i < static_cast<int>(frame.mvbOutlier.size()) && frame.mvbOutlier[i])
            continue;
        ++n_before;

        const int gid = pMP->mnInstanceId;
        if(use_state_suppress)
        {
            if(gid >= 0 && suppressed.find(gid) != suppressed.end())
            {
                idx_state.push_back(i);
                continue;
            }
        }

        // Mask-based removal is a *fallback* channel.
        // When state-based suppression is active, avoid deleting map points already bound to a (non-suppressed)
        // track, since the union dynamic mask can overlap static geometry due to boundary jitter/dilation.
        const bool mask_only_unbound = use_state_suppress;
        const cv::Point2f pt = frame.mvKeys[i].pt;
        bool treat_as_unbound = (gid < 0);
        // If the map point carries an instance id that is not present in the *current* tracker set,
        // treat it as unbound. This keeps the fallback dyn-mask effective under ID churn.
        if(!treat_as_unbound && mask_only_unbound && mpDIFInstanceTracker)
        {
            const auto& trmap = mpDIFInstanceTracker->GetTracks();
            if(trmap.find(gid) == trmap.end())
                treat_as_unbound = true;
        }
        if((!mask_only_unbound || treat_as_unbound) && IsInDIFMask(*dyn_mask, pt))
            idx_mask.push_back(i);
    }

    // 4) 最小约束护栏：walking_rpy 这种剧烈旋转时，RGBD 很容易掉到 <30 inliers
    //    这里用 40 做安全边际（RGBD/Stereo），25 做 IMU 模式安全边际
    int min_keep = 0;
    if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD || mSensor == System::IMU_MONOCULAR)
        min_keep = 25;
    else
        min_keep = 40; // RGBD / Stereo(非IMU) TrackLocalMap 成功门槛是 30

    const int n_after_full =
        n_before - static_cast<int>(idx_state.size()) - static_cast<int>(idx_mask.size());

    const bool skip_mask_removal = (min_keep > 0 && n_after_full < min_keep);

    auto drop_index = [&](int i) {
        frame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        if(i >= 0 && i < static_cast<int>(frame.mvbOutlier.size()))
            frame.mvbOutlier[i] = true;
    };

    // 5) 先执行 state 删除（更可信）
    for(int i : idx_state)
        drop_index(i);
    const int removed_by_state = static_cast<int>(idx_state.size());

    // 6) 如果护栏触发，则跳过 mask 删除（避免硬掉线）
    int removed_by_mask = 0;
    if(!skip_mask_removal)
    {
        for(int i : idx_mask)
            drop_index(i);
        removed_by_mask = static_cast<int>(idx_mask.size());
    }

    // 7) 更详细的日志，便于验证护栏是否在关键帧起作用
    if((removed_by_state + removed_by_mask) > 0 || (skip_mask_removal && !idx_mask.empty()))
    {
        const int fid = static_cast<int>(frame.mnId);
        if(fid != mDIFFilterLastLoggedFrameId)
        {
            mDIFFilterLastLoggedFrameId = fid;
            std::cout << "[DIF-FILTER] frame_id=" << fid
                      << " removed_state=" << removed_by_state
                      << " removed_mask=" << removed_by_mask
                      << " total=" << (removed_by_state + removed_by_mask)
                      << " n_before=" << n_before
                      << " min_keep=" << min_keep
                      << " skip_mask=" << (skip_mask_removal ? 1 : 0)
                      << std::endl;
        }
    }
}

void Tracking::UpdateDIFSuppressedTrackIds()
{
    std::unordered_set<int> next;
    if(mDIFStateCfg.enable && mpDIFInstanceTracker)
    {
        const auto& tracks = mpDIFInstanceTracker->GetTracks();
        next.reserve(tracks.size());
        for(const auto& kv : tracks)
        {
            const DIFTrack& tr = kv.second;
            if(tr.id < 0)
                continue;
            if(tr.state_hat == DIFTrackState::D)
                next.insert(tr.id);
        }
    }
    std::lock_guard<std::mutex> lk(mMutexDIFSuppressed);
    mDIFSuppressedTrackIds.swap(next);
}

void Tracking::UpdateDIFForbiddenForMapTrackIds()
{
    std::unordered_set<int> next;
    if(mDIFStateCfg.enable && mpDIFInstanceTracker)
    {
        const auto& tracks = mpDIFInstanceTracker->GetTracks();
        next.reserve(tracks.size());
        for(const auto& kv : tracks)
        {
            const DIFTrack& tr = kv.second;
            if(tr.id < 0)
                continue;
            // DIF-SLAM v2.0: map-forbid semantics is D only (map_lock must not forbid mapping).
            if(tr.state_hat == DIFTrackState::D)
                next.insert(tr.id);
        }
    }
    std::lock_guard<std::mutex> lk(mMutexDIFForbiddenForMap);
    mDIFForbiddenForMapTrackIds.swap(next);
}

void Tracking::UpdateDIFTrackLockInfoSnapshot()
{
    std::unordered_map<int, DIFTrackLockInfo> next;
    if(mDIFStateCfg.enable && mpDIFInstanceTracker)
    {
        const auto& tracks = mpDIFInstanceTracker->GetTracks();
        next.reserve(tracks.size());
        for(const auto& kv : tracks)
        {
            const DIFTrack& tr = kv.second;
            if(tr.id < 0)
                continue;
            DIFTrackLockInfo info;
            info.map_lock = tr.map_lock;
            info.w_lock = tr.w_lock;
            next.emplace(tr.id, info);
        }
    }
    std::lock_guard<std::mutex> lk(mMutexDIFTrackLockInfo);
    mDIFTrackLockInfo.swap(next);
}

void Tracking::InvalidateDIFPredMasks()
{
    mDIFDynMaskPred.release();
    mDIFDynMaskPredFrameId = -1;
    mDIFDynMaskPredTimestamp = -1.0;
    mDIFForbidMaskPred.release();
    mDIFForbidMaskPredFrameId = -1;
    mDIFForbidMaskPredTimestamp = -1.0;
}

const cv::Mat& Tracking::GetDIFDynamicMaskPredForCurrentFrame()
{
    const int fid = static_cast<int>(mCurrentFrame.mnId);
    const double ts = mCurrentFrame.mTimeStamp;
    if(fid < 0 || mImGray.empty())
    {
        mDIFDynMaskPred.release();
        mDIFDynMaskPredFrameId = -1;
        mDIFDynMaskPredTimestamp = -1.0;
        return mDIFDynMaskPred;
    }
    if(mDIFDynMaskPredFrameId == fid && !mDIFDynMaskPred.empty())
        return mDIFDynMaskPred;
    mDIFDynMaskPred = BuildDIFDynamicMaskFromTracks(mImGray.rows, mImGray.cols, ts);
    mDIFDynMaskPredFrameId = fid;
    mDIFDynMaskPredTimestamp = ts;
    return mDIFDynMaskPred;
}

const cv::Mat& Tracking::GetDIFForbidMaskPredForCurrentFrame()
{
    const int fid = static_cast<int>(mCurrentFrame.mnId);
    const double ts = mCurrentFrame.mTimeStamp;
    if(fid < 0 || mImGray.empty())
    {
        mDIFForbidMaskPred.release();
        mDIFForbidMaskPredFrameId = -1;
        mDIFForbidMaskPredTimestamp = -1.0;
        return mDIFForbidMaskPred;
    }
    if(mDIFForbidMaskPredFrameId == fid && !mDIFForbidMaskPred.empty())
        return mDIFForbidMaskPred;

    const bool has_pose = mCurrentFrame.HasPose();
    const Sophus::SE3f Tcw = has_pose ? mCurrentFrame.GetPose() : Sophus::SE3f();
    mDIFForbidMaskPred = BuildDIFForbidMaskForDense(mImGray.rows, mImGray.cols, ts, has_pose, Tcw);
    mDIFForbidMaskPredFrameId = fid;
    mDIFForbidMaskPredTimestamp = ts;
    return mDIFForbidMaskPred;
}

cv::Mat Tracking::BuildDIFDynamicMaskFromTracks(int height, int width, double timestamp) const
{
    cv::Mat dyn(height, width, CV_8U, cv::Scalar(0));
    if(height <= 0 || width <= 0)
        return dyn;
    if(!mDIFStateCfg.enable || !mpDIFInstanceTracker)
        return dyn;

    const int cur_fid = static_cast<int>(mCurrentFrame.mnId);
    const Sophus::SE3f Tcw = mCurrentFrame.HasPose() ? mCurrentFrame.GetPose() : Sophus::SE3f();
    const bool has_pose = mCurrentFrame.HasPose();
    int max_lag_frames = 0;

    const auto& tracks = mpDIFInstanceTracker->GetTracks();
    for(const auto& kv : tracks)
    {
        const DIFTrack& tr = kv.second;
        if(tr.state_hat != DIFTrackState::D && !(mDIFStateCfg.dyn_mask_include_ever_dynamic && tr.ever_dynamic))
            continue;
        if(tr.mask_last.Empty())
            continue;

        const double dt = (tr.last_timestamp > 0.0 && timestamp > 0.0) ? (timestamp - tr.last_timestamp) : 0.0;
        if(mDIFStateCfg.mask_max_age_s > 0.0f && dt > static_cast<double>(mDIFStateCfg.mask_max_age_s))
            continue;

        // NOTE:
        // - For Module C's M_t^{dyn}, we rely on time-based staleness (mask_max_age_s) + optional warp (mask_warp_*).
        // - Do NOT drop tracks purely by frame-lag here; segmentation can be intentionally low-frequency (every_n_frames>1),
        //   and dropping by lag would disable dynamic filtering almost entirely.
        if(tr.last_frame >= 0 && cur_fid >= 0)
        {
            const int lag_frames = cur_fid - tr.last_frame;
            max_lag_frames = std::max(max_lag_frames, std::max(0, lag_frames));
        }

        const DIFBbox& b = tr.mask_last.bbox;
        if(!b.IsValid())
            continue;

        // Default: no warp.
        int dx = 0;
        int dy = 0;
        bool warped = false;

        if(mDIFStateCfg.mask_warp_enable && tr.has_Cc && tr.has_Tcw && has_pose && mpCamera)
        {
            if(mDIFStateCfg.mask_warp_max_dt <= 0.0f || dt <= static_cast<double>(mDIFStateCfg.mask_warp_max_dt))
            {
                const Sophus::SE3f Trel = Tcw * tr.Tcw_last.inverse(); // c_last -> c_t
                const Eigen::Vector3f Cc_pred = Trel * tr.Cc_last;
                if(Cc_pred.z() > 1e-6f)
                {
                    const Eigen::Vector2f uv = mpCamera->project(Cc_pred);
                    if(!std::isfinite(uv.x()) || !std::isfinite(uv.y()))
                        continue;
                    const cv::Point2f u_pred(uv.x(), uv.y());
                    const cv::Point2f u_last = tr.c2d_last;
                    const cv::Point2f d = u_pred - u_last;
                    const float dn = std::sqrt(d.x * d.x + d.y * d.y);
                    if(!(mDIFStateCfg.mask_warp_max_px > 0.0f) || dn <= mDIFStateCfg.mask_warp_max_px)
                    {
                        dx = static_cast<int>(std::lround(d.x));
                        dy = static_cast<int>(std::lround(d.y));
                        warped = true;
                    }
                }
            }
        }

        DIFBbox dst = b;
        if(warped)
        {
            dst.x1 += dx;
            dst.x2 += dx;
            dst.y1 += dy;
            dst.y2 += dy;
        }

        // Compute copy region with clipping.
        int sx = 0, sy = 0;
        int dx0 = dst.x1, dy0 = dst.y1;
        int copy_w = dst.Width();
        int copy_h = dst.Height();

        if(dx0 < 0)
        {
            sx -= dx0;
            copy_w += dx0;
            dx0 = 0;
        }
        if(dy0 < 0)
        {
            sy -= dy0;
            copy_h += dy0;
            dy0 = 0;
        }
        if(dx0 + copy_w > width)
            copy_w = width - dx0;
        if(dy0 + copy_h > height)
            copy_h = height - dy0;

        if(copy_w <= 0 || copy_h <= 0)
            continue;
        if(sx < 0 || sy < 0)
            continue;
        if(sx + copy_w > tr.mask_last.mask.cols || sy + copy_h > tr.mask_last.mask.rows)
            continue;

        const cv::Mat src = tr.mask_last.mask(cv::Rect(sx, sy, copy_w, copy_h));
        cv::Mat roi = dyn(cv::Rect(dx0, dy0, copy_w, copy_h));
        cv::Mat nz = (src != 0);
        roi.setTo(255, nz);
    }

    int k = std::max(0, mDIFStateCfg.mask_dilate);
    if(mDIFStateCfg.mask_dilate_per_lag > 0 && max_lag_frames > 0)
        k += mDIFStateCfg.mask_dilate_per_lag * max_lag_frames;
    k = std::min(k, 63);

    if(k > 0)
    {
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(std::max(1, k), std::max(1, k)));
        cv::dilate(dyn, dyn, kernel);
    }
    return dyn;
}

cv::Mat Tracking::BuildDIFForbidMaskForDense(int height, int width, double timestamp,
                                             bool has_pose,
                                             const Sophus::SE3f& Tcw) const
{
    cv::Mat forbid(height, width, CV_8U, cv::Scalar(0));
    if(height <= 0 || width <= 0)
        return forbid;
    if(!mDIFStateCfg.enable || !mpDIFInstanceTracker)
        return forbid;

    const auto& tracks = mpDIFInstanceTracker->GetTracks();
    for(const auto& kv : tracks)
    {
        const DIFTrack& tr = kv.second;
        if(tr.id < 0)
            continue;
        // DIF-SLAM v2.0: M_forbid_map == M_D (D pixels only).
        if(tr.state_hat != DIFTrackState::D)
            continue;
        if(tr.mask_last.Empty())
            continue;

        const double dt = (tr.last_timestamp > 0.0 && timestamp > 0.0) ? (timestamp - tr.last_timestamp) : 0.0;
        if(mDIFStateCfg.mask_max_age_s > 0.0f && dt > static_cast<double>(mDIFStateCfg.mask_max_age_s))
            continue;

        int dx = 0;
        int dy = 0;
        bool warped = false;

        if(mDIFStateCfg.mask_warp_enable && tr.has_Cc && tr.has_Tcw && has_pose && mpCamera)
        {
            if(mDIFStateCfg.mask_warp_max_dt <= 0.0f || dt <= static_cast<double>(mDIFStateCfg.mask_warp_max_dt))
            {
                const Sophus::SE3f Trel = Tcw * tr.Tcw_last.inverse();
                const Eigen::Vector3f Cc_pred = Trel * tr.Cc_last;
                if(Cc_pred.z() > 1e-6f)
                {
                    const Eigen::Vector2f uv = mpCamera->project(Cc_pred);
                    if(!std::isfinite(uv.x()) || !std::isfinite(uv.y()))
                        continue;
                    const cv::Point2f u_pred(uv.x(), uv.y());
                    const cv::Point2f u_last = tr.c2d_last;
                    const cv::Point2f d = u_pred - u_last;
                    const float dn = std::sqrt(d.x * d.x + d.y * d.y);
                    if(!(mDIFStateCfg.mask_warp_max_px > 0.0f) || dn <= mDIFStateCfg.mask_warp_max_px)
                    {
                        dx = static_cast<int>(std::lround(d.x));
                        dy = static_cast<int>(std::lround(d.y));
                        warped = true;
                    }
                }
            }
        }

        DIFBbox dst = tr.mask_last.bbox;
        if(warped)
        {
            dst.x1 += dx;
            dst.x2 += dx;
            dst.y1 += dy;
            dst.y2 += dy;
        }

        int sx = 0, sy = 0;
        int dx0 = dst.x1, dy0 = dst.y1;
        int copy_w = dst.Width();
        int copy_h = dst.Height();

        if(dx0 < 0)
        {
            sx -= dx0;
            copy_w += dx0;
            dx0 = 0;
        }
        if(dy0 < 0)
        {
            sy -= dy0;
            copy_h += dy0;
            dy0 = 0;
        }
        if(dx0 + copy_w > width)
            copy_w = width - dx0;
        if(dy0 + copy_h > height)
            copy_h = height - dy0;

        if(copy_w <= 0 || copy_h <= 0)
            continue;
        if(sx < 0 || sy < 0)
            continue;
        if(sx + copy_w > tr.mask_last.mask.cols || sy + copy_h > tr.mask_last.mask.rows)
            continue;

        const cv::Mat src = tr.mask_last.mask(cv::Rect(sx, sy, copy_w, copy_h));
        cv::Mat roi = forbid(cv::Rect(dx0, dy0, copy_w, copy_h));
        cv::Mat nz = (src != 0);
        roi.setTo(255, nz);
    }

    return forbid;
}

void Tracking::ResetDIFRuntimeState(bool reset_instance_tracker)
{
    mDIFDynMaskLast.release();
    mDIFPrevDynMask.release();
    mDIFPrevDynMaskFrameId = -1;
    mDIFLabelMapLast.release();
    mDIFLastSegFrameId = -1;
    mDIFLastSegTimestamp = -1.0;
    mDIFLastDynMaskFrameId = -1;
    mDIFLastDynMaskTimestamp = -1.0;
    mDIFLastPrintSegFrameId = -1;
    mDIFLastPrintBFrameId = -1;
    mDIFLastSegElapsedMs = 0.0;

    mDIFLastBUpdateFrameId = -1;
    mDIFLocalToGlobalFrameId = -1;
    mDIFLocalToGlobal.clear();

    mDIFFrameCache.clear();
    mDIFPendingSegByFrameId.clear();
    mDensePendingKeyFrames.clear();

    {
        std::unique_lock<std::mutex> lock(mMutexDIFSegCache);
        mDIFSegCacheOrder.clear();
    mDIFSegCacheByFrameId.clear();
    }

    mDIFFilterLastLoggedFrameId = -1;

    {
        std::lock_guard<std::mutex> lk(mMutexDIFSuppressed);
        mDIFSuppressedTrackIds.clear();
    }
    {
        std::lock_guard<std::mutex> lk(mMutexDIFForbiddenForMap);
        mDIFForbiddenForMapTrackIds.clear();
    }
    {
        std::lock_guard<std::mutex> lk(mMutexDIFTrackLockInfo);
        mDIFTrackLockInfo.clear();
    }
    InvalidateDIFPredMasks();

    if(reset_instance_tracker && mpDIFInstanceTracker)
        mpDIFInstanceTracker->Reset();

    if(mpDenseMapping)
        mpDenseMapping->Reset();
    mDenseLastMapChangeIdx = -1;
}

void Tracking::PushDIFFrameCache()
{
    if(!mDIFSegCfg.enable)
        return;

    DIFFrameCacheEntry e;
    e.frame_id = static_cast<int>(mCurrentFrame.mnId);
    e.timestamp = mCurrentFrame.mTimeStamp;
    e.has_pose = mCurrentFrame.HasPose();
    if(e.has_pose)
        e.Tcw = mCurrentFrame.GetPose();
    if(!mImRGB.empty())
        e.rgb = mImRGB;
    if(!mImDepth.empty() && mImDepth.type() == CV_32F)
        e.depth = mImDepth;
    e.inliers = (mState == OK) ? mnMatchesInliers : 0;
    e.keypoints_uv.clear();
    e.keypoints_uv.reserve(mCurrentFrame.mvKeys.size());
    for(const auto& kp : mCurrentFrame.mvKeys)
        e.keypoints_uv.push_back(kp.pt);
    e.keypoints_un_uv.clear();
    e.keypoints_un_uv.reserve(mCurrentFrame.mvKeysUn.size());
    for(const auto& kp : mCurrentFrame.mvKeysUn)
        e.keypoints_un_uv.push_back(kp.pt);

    // Cache inlier keypoint matches to the previous frame for feature-3D v_obs.
    e.match_prev_frame_id = -1;
    e.matches_prev.clear();
    if(mLastFrame.isSet() && mCurrentFrame.isSet())
    {
        e.match_prev_frame_id = static_cast<int>(mLastFrame.mnId);
        std::unordered_map<MapPoint*, int> mp_to_prev_idx;
        mp_to_prev_idx.reserve(static_cast<size_t>(std::max(0, mLastFrame.N)));
        for(int i = 0; i < mLastFrame.N; ++i)
        {
            MapPoint* pMP = mLastFrame.mvpMapPoints[static_cast<size_t>(i)];
            if(!pMP)
                continue;
            if(i < static_cast<int>(mLastFrame.mvbOutlier.size()) && mLastFrame.mvbOutlier[static_cast<size_t>(i)])
                continue;
            mp_to_prev_idx.emplace(pMP, i);
        }

        e.matches_prev.reserve(static_cast<size_t>(std::max(0, mCurrentFrame.N)));
        for(int j = 0; j < mCurrentFrame.N; ++j)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[static_cast<size_t>(j)];
            if(!pMP)
                continue;
            if(j < static_cast<int>(mCurrentFrame.mvbOutlier.size()) && mCurrentFrame.mvbOutlier[static_cast<size_t>(j)])
                continue;
            const auto it = mp_to_prev_idx.find(pMP);
            if(it == mp_to_prev_idx.end())
                continue;
            e.matches_prev.emplace_back(it->second, j);
        }
    }

    if(e.frame_id < 0)
        return;

    mDIFFrameCache.push_back(std::move(e));
    while(static_cast<int>(mDIFFrameCache.size()) > std::max(1, mDIFFrameCacheMaxSize))
        mDIFFrameCache.pop_front();

    // Prune pending segmentations that are too old to be matched with cached frames.
    if(!mDIFFrameCache.empty())
    {
        const int oldest = mDIFFrameCache.front().frame_id;
        std::vector<int> to_erase;
        to_erase.reserve(mDIFPendingSegByFrameId.size());
        for(const auto& kv : mDIFPendingSegByFrameId)
        {
            if(kv.first < oldest)
                to_erase.push_back(kv.first);
        }
        for(const int id : to_erase)
            mDIFPendingSegByFrameId.erase(id);
    }
}

void Tracking::StoreDIFPendingSegmentation(const SegmentationResult& seg)
{
    if(!mDIFSegCfg.enable)
        return;
    if(!seg.ok || seg.frame_id < 0 || seg.label_map.empty())
        return;

    // Never enqueue stale results that would force Module-B to run backwards in time.
    if(mDIFLastBUpdateFrameId >= 0 && seg.frame_id <= mDIFLastBUpdateFrameId)
        return;

    mDIFPendingSegByFrameId[seg.frame_id] = seg;
    if(static_cast<int>(mDIFPendingSegByFrameId.size()) > std::max(5, mDIFFrameCacheMaxSize))
    {
        // Best-effort prune: erase the smallest frame_id.
        int min_id = seg.frame_id;
        for(const auto& kv : mDIFPendingSegByFrameId)
            min_id = std::min(min_id, kv.first);
        mDIFPendingSegByFrameId.erase(min_id);
    }
}

void Tracking::ProcessDIFPendingSegmentations()
{
    if(!mDIFSegCfg.enable)
        return;
    if(!mpDIFInstanceTracker)
        return;
    if(mDIFPendingSegByFrameId.empty())
        return;

    std::vector<int> processed;
    processed.reserve(mDIFPendingSegByFrameId.size());

    // IMPORTANT: process pending segmentations in monotonically increasing frame_id order.
    // unordered_map iteration is non-deterministic and can apply newer segmentations before older ones (or apply old
    // ones late), corrupting track continuity and causing global instance IDs to change/swap.
    std::vector<int> fids;
    fids.reserve(mDIFPendingSegByFrameId.size());
    for(const auto& kv : mDIFPendingSegByFrameId)
        fids.push_back(kv.first);
    std::sort(fids.begin(), fids.end());

    for(const int fid : fids)
    {
        auto it = mDIFPendingSegByFrameId.find(fid);
        if(it == mDIFPendingSegByFrameId.end())
            continue;
        SegmentationResult& seg = it->second;
        if(!seg.ok || seg.label_map.empty() || seg.label_map.type() != CV_16S)
            continue;

        // Drop any stale frames that are older than the last successful Module-B update.
        if(mDIFLastBUpdateFrameId >= 0 && fid <= mDIFLastBUpdateFrameId)
        {
            processed.push_back(fid);
            continue;
        }

        const DIFFrameCacheEntry* fe = nullptr;
        for(const auto& e : mDIFFrameCache)
        {
            if(e.frame_id == fid)
            {
                fe = &e;
                break;
            }
        }
        if(!fe || !fe->has_pose || fe->depth.empty() || fe->depth.type() != CV_32F)
            continue;

        cv::Mat label = seg.label_map;
        if(label.rows != fe->depth.rows || label.cols != fe->depth.cols)
        {
            cv::Mat resized;
            cv::resize(label, resized, fe->depth.size(), 0, 0, cv::INTER_NEAREST);
            label = resized;
        }
        // Ensure DenseMapping uses a size-consistent label_map (depth/rgb aligned).
        StoreDIFSegCacheLabelMap(fid, label);

        bool allow_3d_update = true;
        if(mDIFTrackCfg.pose_inliers_min > 0 && fe->inliers < mDIFTrackCfg.pose_inliers_min)
            allow_3d_update = false;
        if(allow_3d_update && mDIFTrackCfg.pose_rbg_max > 0.0f && fe->has_r_bg && std::isfinite(fe->r_bg) &&
           fe->r_bg > mDIFTrackCfg.pose_rbg_max)
            allow_3d_update = false;

        std::vector<int> local_to_global;
        if(mpDIFInstanceTracker->UpdateFromSegmentation(
               fid, seg.timestamp, label, fe->depth, fe->Tcw, mpCamera, allow_3d_update, fe->keypoints_uv, local_to_global))
        {
            if(fid > mDIFLastBUpdateFrameId)
                mDIFLastBUpdateFrameId = fid;
            // Keep the most recent mapping for optional MapPoint binding in Tracking.
            mDIFLocalToGlobalFrameId = fid;
            mDIFLocalToGlobal = std::move(local_to_global);
            StoreDIFSegCacheLocalToGlobal(fid, mDIFLocalToGlobal);
            // Keyframes created earlier in this frame may have deferred dense integration because local_to_global was
            // not ready yet. Try to flush now.
            MaybeFlushPendingDenseKeyFrame(fid);
            MaybeEnqueueDenseMappingFromSegFrame(fid, seg.timestamp, label, *fe, mDIFLocalToGlobal);
            UpdateDIFTrackVobsFromFeature3D(fid, label, *fe, mDIFLocalToGlobal, allow_3d_update);

            if(fid != mDIFLastPrintBFrameId)
            {
                mDIFLastPrintBFrameId = fid;
                int n_has3d = 0;
                int n_vobs = 0;
                for(const auto& kv2 : mpDIFInstanceTracker->GetTracks())
                {
                    const DIFTrack& tr = kv2.second;
                    if(tr.has_Cc)
                        n_has3d++;
                    if(tr.last_v_obs_frame == fid)
                        n_vobs++;
                }
                std::cout << "[DIF-B] updated tracks at frame_id=" << fid
                          << " tracks=" << mpDIFInstanceTracker->GetTracks().size()
                          << " has3d=" << n_has3d
                          << " vobs=" << n_vobs
                          << " allow_3d=" << (allow_3d_update ? 1 : 0)
                          << " inliers=" << fe->inliers
                          << " r_bg=" << (fe->has_r_bg ? fe->r_bg : -1.0f)
                          << std::endl;
            }
        }

        processed.push_back(fid);
    }

    for(const int fid : processed)
        mDIFPendingSegByFrameId.erase(fid);
}

void Tracking::UpdateDIFTrackVobsFromFeature3D(
    int frame_id,
    const cv::Mat& label_map,
    const DIFFrameCacheEntry& fe,
    const std::vector<int>& local_to_global,
    bool allow_3d_update)
{
    if(!mDIFSegCfg.enable || !mpDIFInstanceTracker)
        return;
    if(mDIFTrackCfg.vobs_mode != 1 && mDIFTrackCfg.vobs_mode != 2)
        return;
    if(!allow_3d_update)
        return;
    if(label_map.empty() || label_map.type() != CV_16S)
        return;
    if(local_to_global.empty())
        return;
    if(fe.match_prev_frame_id < 0 || fe.matches_prev.empty())
        return;

    const DIFFrameCacheEntry* fprev = nullptr;
    for(const auto& e : mDIFFrameCache)
    {
        if(e.frame_id == fe.match_prev_frame_id)
        {
            fprev = &e;
            break;
        }
    }
    if(!fprev)
        return;
    if(!fe.has_pose || !fprev->has_pose)
        return;
    if(fe.depth.empty() || fe.depth.type() != CV_32F || fprev->depth.empty() || fprev->depth.type() != CV_32F)
        return;
    if(fe.keypoints_uv.empty() || fe.keypoints_un_uv.empty() || fprev->keypoints_uv.empty() || fprev->keypoints_un_uv.empty())
        return;

    const double dt = fe.timestamp - fprev->timestamp;
    if(!(dt > 1e-6))
        return;
    if(mDIFTrackCfg.vobs_dt_min > 0.0f && dt < static_cast<double>(mDIFTrackCfg.vobs_dt_min))
        return;

    const float fx = mpCamera ? mpCamera->getParameter(0) : 0.f;
    const float fy = mpCamera ? mpCamera->getParameter(1) : 0.f;
    const float cx = mpCamera ? mpCamera->getParameter(2) : 0.f;
    const float cy = mpCamera ? mpCamera->getParameter(3) : 0.f;
    if(!(fx > 1e-6f) || !(fy > 1e-6f))
        return;
    const float invfx = 1.0f / fx;
    const float invfy = 1.0f / fy;
    const float f_avg = 0.5f * (fx + fy);

    cv::Mat prev_label_map;
    std::vector<int> prev_local_to_global;
    bool has_prev_label = false;
    if(mDIFTrackCfg.vobs_bi_label_enable)
    {
        std::unique_lock<std::mutex> lock(mMutexDIFSegCache);
        const auto itp = mDIFSegCacheByFrameId.find(fe.match_prev_frame_id);
        if(itp != mDIFSegCacheByFrameId.end())
        {
            prev_label_map = itp->second.label_map;
            prev_local_to_global = itp->second.local_to_global;
        }
        has_prev_label =
            (!prev_label_map.empty() && prev_label_map.type() == CV_16S && !prev_local_to_global.empty());
    }

    auto clamp01 = [](float x) -> float { return std::max(0.0f, std::min(1.0f, x)); };

    auto median_in_place = [](std::vector<float>& v) -> float {
        if(v.empty())
            return 0.0f;
        const size_t k = v.size() / 2;
        const auto it = v.begin() + static_cast<std::vector<float>::difference_type>(k);
        std::nth_element(v.begin(), it, v.end());
        return *it;
    };

    auto depth_at = [](const cv::Mat& depth, const cv::Point2f& uv) -> float {
        const int x = static_cast<int>(uv.x);
        const int y = static_cast<int>(uv.y);
        if(x < 0 || y < 0 || x >= depth.cols || y >= depth.rows)
            return -1.0f;
        const float z = depth.at<float>(y, x);
        if(!(z > 0.0f) || !std::isfinite(z))
            return -1.0f;
        return z;
    };

    auto backproject = [&](const cv::Point2f& uv_un, float z) -> Eigen::Vector3f {
        return Eigen::Vector3f((uv_un.x - cx) * z * invfx, (uv_un.y - cy) * z * invfy, z);
    };

    auto is_inside_eroded = [&](int x, int y, int local_id) -> bool {
        const int r = std::max(0, mDIFTrackCfg.vobs_mask_erode_px);
        if(r <= 0)
            return true;
        if(x - r < 0 || y - r < 0 || x + r >= label_map.cols || y + r >= label_map.rows)
            return false;
        for(int dy = -r; dy <= r; ++dy)
        {
            const int16_t* row = label_map.ptr<int16_t>(y + dy);
            for(int dx = -r; dx <= r; ++dx)
            {
                if(row[x + dx] != local_id)
                    return false;
            }
        }
        return true;
    };

    auto gid_at_prev = [&](const cv::Point2f& pt) -> int {
        if(!has_prev_label)
            return -1;
        const int x = static_cast<int>(pt.x + 0.5f);
        const int y = static_cast<int>(pt.y + 0.5f);
        if(x < 0 || y < 0 || x >= prev_label_map.cols || y >= prev_label_map.rows)
            return -1;
        const int local_id = static_cast<int>(prev_label_map.at<int16_t>(y, x));
        if(local_id < 0 || local_id >= static_cast<int>(prev_local_to_global.size()))
            return -1;
        return prev_local_to_global[static_cast<size_t>(local_id)];
    };

    struct ResidualSample
    {
        float r; // meters
        float z; // meters (current frame depth)

        ResidualSample() : r(0.0f), z(0.0f) {}
        ResidualSample(float r_, float z_) : r(r_), z(z_) {}
    };

    std::unordered_map<int, std::vector<ResidualSample>> buckets;
    std::unordered_map<int, int> raw_counts;
    buckets.reserve(64);
    raw_counts.reserve(64);

    const Sophus::SE3f Trel = fe.Tcw * fprev->Tcw.inverse(); // c_prev -> c_cur

    for(const auto& m : fe.matches_prev)
    {
        const int i_prev = m.first;
        const int i_cur = m.second;
        if(i_prev < 0 || i_cur < 0)
            continue;
        if(i_prev >= static_cast<int>(fprev->keypoints_uv.size()) || i_cur >= static_cast<int>(fe.keypoints_uv.size()))
            continue;
        if(i_prev >= static_cast<int>(fprev->keypoints_un_uv.size()) || i_cur >= static_cast<int>(fe.keypoints_un_uv.size()))
            continue;

        const cv::Point2f uv_cur = fe.keypoints_uv[static_cast<size_t>(i_cur)];
        const int x = static_cast<int>(uv_cur.x + 0.5f);
        const int y = static_cast<int>(uv_cur.y + 0.5f);
        if(x < 0 || y < 0 || x >= label_map.cols || y >= label_map.rows)
            continue;
        const int local_id = static_cast<int>(label_map.at<int16_t>(y, x));
        if(local_id < 0 || local_id >= static_cast<int>(local_to_global.size()))
            continue;
        const int gid = local_to_global[static_cast<size_t>(local_id)];
        if(gid < 0)
            continue;

        raw_counts[gid]++;

        if(!is_inside_eroded(x, y, local_id))
            continue;

        if(has_prev_label)
        {
            const cv::Point2f uv_prev = fprev->keypoints_uv[static_cast<size_t>(i_prev)];
            const int gid_prev = gid_at_prev(uv_prev);
            if(gid_prev != gid)
                continue;
        }

        const float z0 = depth_at(fprev->depth, fprev->keypoints_uv[static_cast<size_t>(i_prev)]);
        const float z1 = depth_at(fe.depth, fe.keypoints_uv[static_cast<size_t>(i_cur)]);
        if(!(z0 > 0.0f) || !(z1 > 0.0f))
            continue;

        if(mDIFTrackCfg.vobs_depth_jump_ratio_max > 0.0f)
        {
            const float eps = 1e-6f;
            const float rz = std::abs(z1 - z0) / std::max(z0, eps);
            if(rz > mDIFTrackCfg.vobs_depth_jump_ratio_max)
                continue;
        }

        const Eigen::Vector3f Xprev = backproject(fprev->keypoints_un_uv[static_cast<size_t>(i_prev)], z0);
        const Eigen::Vector3f Xcur = backproject(fe.keypoints_un_uv[static_cast<size_t>(i_cur)], z1);
        const Eigen::Vector3f Xhat = Trel * Xprev;
        const float r = (Xcur - Xhat).norm();
        if(!std::isfinite(r))
            continue;

        auto& v = buckets[gid];
        if(mDIFTrackCfg.vobs_match_n_max > 0 && static_cast<int>(v.size()) >= mDIFTrackCfg.vobs_match_n_max)
            continue;
        v.emplace_back(r, z1);
    }

    auto& tracks = mpDIFInstanceTracker->GetTracksMutable();
    const int image_area = label_map.rows * label_map.cols;
    const float dt_f = static_cast<float>(dt);

    for(auto& kv : buckets)
    {
        const int gid = kv.first;
        std::vector<ResidualSample>& samples = kv.second;
        const int n_raw = static_cast<int>(samples.size());
        if(n_raw < std::max(1, mDIFTrackCfg.vobs_match_n_min))
            continue;

        if(mDIFTrackCfg.vobs_edge_ratio_min > 0.0f)
        {
            const auto itc = raw_counts.find(gid);
            if(itc != raw_counts.end() && itc->second > 0)
            {
                const float ratio = static_cast<float>(n_raw) / static_cast<float>(itc->second);
                if(ratio < mDIFTrackCfg.vobs_edge_ratio_min)
                    continue;
            }
        }

        auto it_tr = tracks.find(gid);
        if(it_tr == tracks.end())
            continue;
        DIFTrack& tr = it_tr->second;

        std::vector<float> rs;
        rs.reserve(samples.size());
        for(const auto& s : samples)
            rs.push_back(s.r);

        std::vector<float> tmp = rs;
        const float r_med = median_in_place(tmp);

        std::vector<float> abs_dev;
        abs_dev.reserve(rs.size());
        for(const float r : rs)
            abs_dev.push_back(std::abs(r - r_med));
        const float mad = median_in_place(abs_dev);
        const float sigma_rob = 1.4826f * std::max(mad, 1e-6f);
        if(mDIFTrackCfg.vobs_sigma_r_max > 0.0f && sigma_rob > mDIFTrackCfg.vobs_sigma_r_max)
            continue;

        const float kappa = std::max(0.0f, mDIFTrackCfg.vobs_mad_kappa);
        const float thr = r_med + kappa * sigma_rob;

        std::vector<float> inliers_r;
        std::vector<float> inliers_z;
        inliers_r.reserve(samples.size());
        inliers_z.reserve(samples.size());
        for(const auto& s : samples)
        {
            if(s.r <= thr)
            {
                inliers_r.push_back(s.r);
                inliers_z.push_back(s.z);
            }
        }

        const int n_in = static_cast<int>(inliers_r.size());
        if(n_in < std::max(1, mDIFTrackCfg.vobs_match_n_min))
            continue;
        if(mDIFTrackCfg.vobs_inlier_ratio_min > 0.0f)
        {
            const float ratio = static_cast<float>(n_in) / static_cast<float>(n_raw);
            if(ratio < mDIFTrackCfg.vobs_inlier_ratio_min)
                continue;
        }

        std::sort(inliers_r.begin(), inliers_r.end());
        const float beta = std::max(0.0f, std::min(0.49f, mDIFTrackCfg.vobs_trim_frac));
        int trim = static_cast<int>(std::floor(beta * static_cast<float>(inliers_r.size())));
        trim = std::max(0, std::min(trim, static_cast<int>(inliers_r.size()) / 2));

        float delta = 0.0f;
        {
            const int i0 = trim;
            const int i1 = static_cast<int>(inliers_r.size()) - trim;
            const int n = std::max(0, i1 - i0);
            if(n <= 0)
            {
                delta = r_med;
            }
            else
            {
                double sum = 0.0;
                for(int i = i0; i < i1; ++i)
                    sum += static_cast<double>(inliers_r[static_cast<size_t>(i)]);
                delta = static_cast<float>(sum / static_cast<double>(n));
            }
        }

        float delta_bias = 0.0f;
        if(f_avg > 1e-6f && (mDIFTrackCfg.vobs_px_sigma > 0.0f || mDIFTrackCfg.vobs_depth_sigma_m > 0.0f))
        {
            std::vector<float> biases;
            biases.reserve(inliers_z.size());
            for(const float z : inliers_z)
            {
                if(!(z > 0.0f) || !std::isfinite(z))
                    continue;
                const float b_px =
                    (mDIFTrackCfg.vobs_px_sigma > 0.0f) ? (std::sqrt(2.0f) * z * mDIFTrackCfg.vobs_px_sigma / f_avg) : 0.0f;
                const float b_z =
                    (mDIFTrackCfg.vobs_depth_sigma_m > 0.0f) ? (std::sqrt(2.0f) * mDIFTrackCfg.vobs_depth_sigma_m) : 0.0f;
                biases.push_back(std::sqrt(b_px * b_px + b_z * b_z));
            }
            if(!biases.empty())
                delta_bias = median_in_place(biases);
        }

        const float delta_eff = std::max(0.0f, delta - delta_bias);
        const float v_meas = delta_eff / std::max(dt_f, 1e-6f);

        // Optional background-like suppression (large planar instances).
        bool suppress_v = false;
        if(mDIFTrackCfg.vobs_bg_suppress_enable && image_area > 0 && mDIFTrackCfg.vobs_bg_area_ratio_min > 0.0f &&
           mDIFTrackCfg.vobs_bg_z_sigma_max > 0.0f)
        {
            const float area_ratio = static_cast<float>(tr.mask_last.area) / static_cast<float>(image_area);
            if(area_ratio >= mDIFTrackCfg.vobs_bg_area_ratio_min)
            {
                std::vector<float> ztmp = inliers_z;
                const float z_med = median_in_place(ztmp);
                std::vector<float> zdev;
                zdev.reserve(ztmp.size());
                for(const float z : ztmp)
                    zdev.push_back(std::abs(z - z_med));
                const float z_mad = median_in_place(zdev);
                const float z_sigma_rob = 1.4826f * std::max(z_mad, 1e-6f);
                const bool bg_like = (z_sigma_rob <= mDIFTrackCfg.vobs_bg_z_sigma_max);
                if(bg_like && v_meas < mDIFTrackCfg.vobs_bg_v_min)
                    suppress_v = true;
            }
        }
        if(suppress_v)
            continue;

        const float q_n =
            (mDIFTrackCfg.vobs_q_n_ref > 0) ? (static_cast<float>(n_in) / static_cast<float>(mDIFTrackCfg.vobs_q_n_ref)) : 1.0f;
        const float q_s =
            (mDIFTrackCfg.vobs_q_sigma_ref > 0.0f) ? (mDIFTrackCfg.vobs_q_sigma_ref / (sigma_rob + 1e-6f)) : 1.0f;
        const float q_v = clamp01(q_n) * clamp01(q_s);
        if(q_v < std::max(0.0f, mDIFTrackCfg.vobs_q_min))
            continue;

        const float beta_ema = clamp01(mDIFTrackCfg.vel_ema_beta);
        const bool had_v = (tr.last_v_obs_used_frame >= 0) || (tr.last_v_obs_frame >= 0);
        tr.v_obs = had_v ? ((1.0f - beta_ema) * tr.v_obs + beta_ema * v_meas) : v_meas;
        tr.v_obs_q = q_v;
        tr.last_v_obs_frame = frame_id;
    }
}

int Tracking::GetDIFGlobalTrackIdAt(const cv::Point2f& pt) const
{
    const int cur_id = static_cast<int>(mCurrentFrame.mnId);
    if(mDIFLocalToGlobalFrameId != cur_id)
        return -1;
    if(mDIFLastSegFrameId != cur_id)
        return -1;
    if(mDIFLabelMapLast.empty() || mDIFLabelMapLast.type() != CV_16S)
        return -1;
    const int x = static_cast<int>(pt.x + 0.5f);
    const int y = static_cast<int>(pt.y + 0.5f);
    if(x < 0 || y < 0 || x >= mDIFLabelMapLast.cols || y >= mDIFLabelMapLast.rows)
        return -1;
    const int local_id = static_cast<int>(mDIFLabelMapLast.at<int16_t>(y, x));
    if(local_id < 0 || local_id >= static_cast<int>(mDIFLocalToGlobal.size()))
        return -1;
    return mDIFLocalToGlobal[static_cast<size_t>(local_id)];
}

int Tracking::GetDIFGlobalTrackIdAtFrameId(int frame_id, const cv::Point2f& pt) const
{
    if(frame_id < 0)
        return -1;

    std::unique_lock<std::mutex> lock(mMutexDIFSegCache);
    const auto it = mDIFSegCacheByFrameId.find(frame_id);
    if(it == mDIFSegCacheByFrameId.end())
        return -1;
    const DIFSegCacheEntry& e = it->second;
    if(e.label_map.empty() || e.label_map.type() != CV_16S)
        return -1;
    if(e.local_to_global.empty())
        return -1;

    const int x = static_cast<int>(pt.x + 0.5f);
    const int y = static_cast<int>(pt.y + 0.5f);
    if(x < 0 || y < 0 || x >= e.label_map.cols || y >= e.label_map.rows)
        return -1;

    const int local_id = static_cast<int>(e.label_map.at<int16_t>(y, x));
    if(local_id < 0 || local_id >= static_cast<int>(e.local_to_global.size()))
        return -1;
    return e.local_to_global[static_cast<size_t>(local_id)];
}

bool Tracking::HasDIFSegMappingForFrameId(int frame_id) const
{
    if(frame_id < 0)
        return false;
    std::unique_lock<std::mutex> lock(mMutexDIFSegCache);
    const auto it = mDIFSegCacheByFrameId.find(frame_id);
    if(it == mDIFSegCacheByFrameId.end())
        return false;
    const DIFSegCacheEntry& e = it->second;
    return (!e.label_map.empty() && e.label_map.type() == CV_16S && !e.local_to_global.empty());
}

void Tracking::StoreDIFSegCacheLabelMap(int frame_id, const cv::Mat& label_map)
{
    if(frame_id < 0)
        return;
    if(label_map.empty() || label_map.type() != CV_16S)
        return;

    std::unique_lock<std::mutex> lock(mMutexDIFSegCache);
    DIFSegCacheEntry& e = mDIFSegCacheByFrameId[frame_id];
    e.label_map = label_map;

    // LRU-ish order maintenance.
    {
        auto it = std::find(mDIFSegCacheOrder.begin(), mDIFSegCacheOrder.end(), frame_id);
        if(it != mDIFSegCacheOrder.end())
            mDIFSegCacheOrder.erase(it);
    }
    mDIFSegCacheOrder.push_back(frame_id);
    while(static_cast<int>(mDIFSegCacheOrder.size()) > std::max(1, mDIFSegCacheMaxSize))
    {
        const int oldest = mDIFSegCacheOrder.front();
        mDIFSegCacheOrder.pop_front();
        mDIFSegCacheByFrameId.erase(oldest);
    }
}

void Tracking::StoreDIFSegCacheLocalToGlobal(int frame_id, const std::vector<int>& local_to_global)
{
    if(frame_id < 0)
        return;
    if(local_to_global.empty())
        return;

    std::unique_lock<std::mutex> lock(mMutexDIFSegCache);
    DIFSegCacheEntry& e = mDIFSegCacheByFrameId[frame_id];
    e.local_to_global = local_to_global;

    {
        auto it = std::find(mDIFSegCacheOrder.begin(), mDIFSegCacheOrder.end(), frame_id);
        if(it != mDIFSegCacheOrder.end())
            mDIFSegCacheOrder.erase(it);
    }
    mDIFSegCacheOrder.push_back(frame_id);
    while(static_cast<int>(mDIFSegCacheOrder.size()) > std::max(1, mDIFSegCacheMaxSize))
    {
        const int oldest = mDIFSegCacheOrder.front();
        mDIFSegCacheOrder.pop_front();
        mDIFSegCacheByFrameId.erase(oldest);
    }
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}



Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    //cout << "GrabImageStereo" << endl;

    mImGrayPrev = mImGray;
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        //cout << "Image with 3 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        //cout << "Image with 4 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    //cout << "Incoming frame creation" << endl;

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    //cout << "Incoming frame ended" << endl;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    //cout << "Tracking start" << endl;
    Track();
    //cout << "Tracking end" << endl;

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    mImRGB = imRGB;
    mImGrayPrev = mImGray;
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    // Keep depth for DIF (Module B 3D centroid sampling)
    mImDepth = imDepth;

    if (mSensor == System::RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::IMU_RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);






    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    Track();

    MaybeRebuildDenseMappingAfterLoop();

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mImGrayPrev = mImGray;
    mImGray = im;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if (mSensor == System::MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
    }
    else if(mSensor == System::IMU_MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }

    if (mState==NO_IMAGES_YET)
        t0=timestamp;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId;
    Track();

    return mCurrentFrame.GetPose();
}


void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{

    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mCurrentFrame.mTimeStamp-mImuPer)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }

    const int n = mvImuFromLastFrame.size()-1;
    if(n==0){
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);

    for(int i=0; i<n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    mCurrentFrame.setIntegrated();

    //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}


bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    if(mbMapUpdated && mpLastKeyFrame)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else if(!mbMapUpdated)
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}


void Tracking::Track()
{

    if (bStepByStep)
    {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        while(!mbStep && bStepByStep)
            usleep(500);
        mbStep = false;
    }

    if(mDIFSegCfg.enable && mpDIFSegWorker)
    {
        const int cur_id = static_cast<int>(mCurrentFrame.mnId);
        const bool do_submit = (mDIFSegCfg.every_n_frames > 0) && ((cur_id % mDIFSegCfg.every_n_frames) == 0);
        if(do_submit)
        {
            mpDIFSegWorker->SubmitFrame(cur_id, mCurrentFrame.mTimeStamp, mImRGB, mbRGB);
            const bool initial_sync_wait = (mDIFInitialSync && !mDIFInitialSyncDone && mDIFInitialSyncTimeoutMs > 0);
            const bool regular_sync_wait = (mDIFSync && mDIFSyncTimeoutMs > 0);
            const int wait_ms = initial_sync_wait ? mDIFInitialSyncTimeoutMs : (regular_sync_wait ? mDIFSyncTimeoutMs : 0);
            if(wait_ms > 0)
            {
                SegmentationResult seg;
                if(mpDIFSegWorker->WaitForFrameResult(cur_id, wait_ms, seg))
                {
                    // Reuse the same consume logic (prints + caches + debug mask)
                    if(seg.frame_id != mDIFLastPrintSegFrameId)
                    {
                        mDIFLastPrintSegFrameId = seg.frame_id;
                        const int lag = (seg.frame_id >= 0) ? (cur_id - seg.frame_id) : -1;
                        // Frame separator for better log readability
                        std::cout << "------------------------ [DIF Frame " << seg.frame_id << "] ------------------------" << std::endl;
                        if(seg.ok && !seg.label_map.empty())
                        {
                            std::cout << "[DIF] seg ok frame_id=" << seg.frame_id
                                      << " cur=" << cur_id
                                      << " lag=" << lag
                                      << " elapsed_ms=" << seg.elapsed_ms
                                      << " infer_ms=" << seg.profile_infer_ms
                                      << " post_ms=" << seg.profile_post_ms
                                      << " label=" << seg.label_map.cols << "x" << seg.label_map.rows
                                      << " debug_mask_mode=" << mDIFDebugMaskMode
                                      << " filter_enable=" << (mDIFFilterEnable ? 1 : 0)
                                      << std::endl;
                        }
                        else
                        {
                            std::cout << "[DIF] seg failed frame_id=" << seg.frame_id
                                      << " cur=" << cur_id
                                      << " lag=" << lag
                                      << " elapsed_ms=" << seg.elapsed_ms
                                      << " error=" << seg.error
                                      << std::endl;
                        }
                    }
                    if(seg.ok && !seg.label_map.empty())
                        UpdateDIFDebugMaskFromLabelMap(seg);
                }
                if(initial_sync_wait)
                    mDIFInitialSyncDone = true;
            }
        }
        if(!mDIFSync)
            UpdateDIFSegmentation();
    }

    if(mpLocalMapper->mbBadImu)
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }

    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if(!pCurrentMap)
    {
        cout << "ERROR: There is not an active map in the atlas" << endl;
    }

    if(mState!=NO_IMAGES_YET)
    {
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
        {
            // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
            // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            if(mpAtlas->isInertial())
            {

                if(mpAtlas->isImuInitialized())
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        mpSystem->ResetActiveMap();
                    }
                    else
                    {
                        CreateMapInAtlas();
                    }
                }
                else
                {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap();
                }
                return;
            }

        }
    }


    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU();
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu);
#endif

    }
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false;

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }


    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
        {
            StereoInitialization();
        }
        else
        {
            MonocularInitialization();
        }

        //mpFrameDrawer->Update(this);

        if(mState!=OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        if(mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(mState==OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }


                if (!bOK)
                {
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST;
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10)
                    {
                        // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                    }
                    else
                    {
                        mState = LOST;
                    }
                }
            }
            else
            {

                if (mState == RECENTLY_LOST)
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true;
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                    else
                    {
                        // Relocalization
                        bOK = Relocalization();
                        //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        if(mCurrentFrame.mTimeStamp-mTimeStampLost>3.0f && !bOK)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST)
                {

                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap()<10)
                    {
                        mpSystem->ResetActiveMap();
                        Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    }else
                        CreateMapInAtlas();

                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            }

        }
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if(mState==LOST)
            {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(mbVelocity)
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    Sophus::SE3f TcwMM;
                    if(mbVelocity)
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.GetPose();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred);
#endif


#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                bOK = TrackLocalMap();

            }
            if(!bOK)
                cout << "Fail to track local map!" << endl;
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else if (mState == OK)
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState=RECENTLY_LOST;
            }
            else
                mState=RECENTLY_LOST; // visual to lost

            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            //}
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
           (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
        {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if(pCurrentMap->isImuInitialized())
        {
            if(bOK)
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU();
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    mLastBias = mCurrentFrame.mImuBias;
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        // DIF Module B/C update (produces M_t^{dyn} for next-frame filtering)
        if(mDIFSegCfg.enable)
        {
            PushDIFFrameCache();
            ProcessDIFPendingSegmentations();

            if(mpDIFStateEstimator && mpDIFInstanceTracker)
            {
                const int fid_cur = static_cast<int>(mCurrentFrame.mnId);

                const cv::Mat* label_map = nullptr;
                const std::vector<int>* local_to_global = nullptr;
                if(mDIFLastSegFrameId == fid_cur && mDIFLocalToGlobalFrameId == fid_cur && !mDIFLabelMapLast.empty() &&
                   mDIFLabelMapLast.type() == CV_16S && !mDIFLocalToGlobal.empty())
                {
                    label_map = &mDIFLabelMapLast;
                    local_to_global = &mDIFLocalToGlobal;
                }

	                // Frontend rigid-flow residuals (used as an additional circuit-breaker channel).
	                std::vector<float> flow_bg;
	                std::unordered_map<int, std::vector<float>> flow_inst;
	                const bool has_depth = !mLastFrame.mvDepth.empty();
	                const bool has_desc = !mLastFrame.mDescriptors.empty() && !mCurrentFrame.mDescriptors.empty();
	                const bool has_lk_img = !mImGrayPrev.empty() && !mImGray.empty() && mImGrayPrev.type() == CV_8U &&
	                                        mImGray.type() == CV_8U && mImGrayPrev.size() == mImGray.size();
	                if(mDIFStateCfg.flow_enable && label_map && local_to_global && mLastFrame.isSet() && mCurrentFrame.isSet() &&
	                   mLastFrame.HasPose() && mCurrentFrame.HasPose() && has_depth && mpCamera && (has_desc || has_lk_img))
	                {
	                    // Also try to attribute flow residuals to instances using the previous frame's segmentation cache.
	                    // This significantly improves recall for moving instances, where a matched keypoint may fall outside
	                    // the current mask due to motion/segmentation boundary jitter (otherwise it gets counted as bg and
                    // dilutes the instance residual statistics).
                    cv::Mat prev_label_map;
                    std::vector<int> prev_local_to_global;
                    {
                        const int fid_prev = static_cast<int>(mLastFrame.mnId);
                        std::unique_lock<std::mutex> lock(mMutexDIFSegCache);
                        const auto itp = mDIFSegCacheByFrameId.find(fid_prev);
                        if(itp != mDIFSegCacheByFrameId.end())
                        {
                            prev_label_map = itp->second.label_map;
                            prev_local_to_global = itp->second.local_to_global;
                        }
                    }
                    auto gid_at_prev = [&](const cv::Point2f& pt) -> int {
                        if(prev_label_map.empty() || prev_label_map.type() != CV_16S || prev_local_to_global.empty())
                            return -1;
                        const int x = static_cast<int>(pt.x + 0.5f);
                        const int y = static_cast<int>(pt.y + 0.5f);
                        if(x < 0 || y < 0 || x >= prev_label_map.cols || y >= prev_label_map.rows)
                            return -1;
                        const int local_id = static_cast<int>(prev_label_map.at<int16_t>(y, x));
                        if(local_id < 0 || local_id >= static_cast<int>(prev_local_to_global.size()))
                            return -1;
                        return prev_local_to_global[static_cast<size_t>(local_id)];
                    };

	                    const Sophus::SE3f Tcw_prev = mLastFrame.GetPose();
	                    const Sophus::SE3f Tcw_cur = mCurrentFrame.GetPose();
	                    const Sophus::SE3f Trel = Tcw_cur * Tcw_prev.inverse(); // c_{t-1} -> c_t

	                    constexpr int kMaxMatches = 2000;

	                    auto compute_flow_desc = [&]() -> int {
	                        if(!has_desc)
	                            return 0;

	                        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
	                        std::vector<std::vector<cv::DMatch>> knn;
	                        matcher.knnMatch(mLastFrame.mDescriptors, mCurrentFrame.mDescriptors, knn, 2);

	                        constexpr float kRatio = 0.80f;
	                        int kept = 0;
	                        for(const auto& vv : knn)
	                        {
	                            if(static_cast<int>(vv.size()) < 2)
	                                continue;
	                            const cv::DMatch& m0 = vv[0];
	                            const cv::DMatch& m1 = vv[1];
	                            if(m0.distance >= kRatio * m1.distance)
	                                continue;
	                            const int i_prev = m0.queryIdx;
	                            const int i_cur = m0.trainIdx;
	                            if(i_prev < 0 || i_cur < 0)
	                                continue;
	                            if(i_prev >= static_cast<int>(mLastFrame.mvKeysUn.size()) ||
	                               i_cur >= static_cast<int>(mCurrentFrame.mvKeysUn.size()))
	                                continue;
	                            if(i_prev >= static_cast<int>(mLastFrame.mvDepth.size()))
	                                continue;

	                            const float z = mLastFrame.mvDepth[static_cast<size_t>(i_prev)];
	                            if(!(z > 0.0f) || !std::isfinite(z))
	                                continue;

	                            const cv::Point2f uv_prev = mLastFrame.mvKeysUn[static_cast<size_t>(i_prev)].pt;
	                            const Eigen::Vector3f Xprev(
	                                (uv_prev.x - Frame::cx) * z * Frame::invfx,
	                                (uv_prev.y - Frame::cy) * z * Frame::invfy,
	                                z);
	                            const Eigen::Vector3f Xcur = Trel * Xprev;
	                            if(!(Xcur.z() > 1e-6f))
	                                continue;

	                            const Eigen::Vector2f uv_hat = mpCamera->project(Xcur);
	                            if(!std::isfinite(uv_hat.x()) || !std::isfinite(uv_hat.y()))
	                                continue;

	                            const cv::Point2f uv_cur = mCurrentFrame.mvKeysUn[static_cast<size_t>(i_cur)].pt;
	                            const float dx = uv_cur.x - uv_hat.x();
	                            const float dy = uv_cur.y - uv_hat.y();
	                            const float e = std::sqrt(dx * dx + dy * dy);
	                            if(!std::isfinite(e))
	                                continue;

	                            int gid = GetDIFGlobalTrackIdAt(uv_cur);
	                            if(gid < 0)
	                                gid = gid_at_prev(uv_prev);
	                            if(gid >= 0)
	                                flow_inst[gid].push_back(e);
	                            else
	                                flow_bg.push_back(e);

	                            if(++kept >= kMaxMatches)
	                                break;
	                        }
	                        return kept;
	                    };

	                    auto compute_flow_lk = [&]() -> int {
	                        if(!has_lk_img)
	                            return 0;
	                        // LK tracking is performed in the pixel domain; only enable when distortion is negligible.
	                        if(!mDistCoef.empty() && cv::norm(mDistCoef, cv::NORM_INF) > 1e-12)
	                            return 0;

	                        std::vector<cv::Point2f> prev_pts;
	                        std::vector<cv::Point2f> next_pts;
	                        std::vector<cv::Point2f> hat_pts;
	                        prev_pts.reserve(kMaxMatches);
	                        next_pts.reserve(kMaxMatches);
	                        hat_pts.reserve(kMaxMatches);

	                        const int n_prev = static_cast<int>(mLastFrame.mvKeysUn.size());
	                        for(int i_prev = 0; i_prev < n_prev && static_cast<int>(prev_pts.size()) < kMaxMatches; ++i_prev)
	                        {
	                            if(i_prev >= static_cast<int>(mLastFrame.mvDepth.size()))
	                                break;
	                            const float z = mLastFrame.mvDepth[static_cast<size_t>(i_prev)];
	                            if(!(z > 0.0f) || !std::isfinite(z))
	                                continue;

	                            const cv::Point2f uv_prev = mLastFrame.mvKeysUn[static_cast<size_t>(i_prev)].pt;
	                            const Eigen::Vector3f Xprev(
	                                (uv_prev.x - Frame::cx) * z * Frame::invfx,
	                                (uv_prev.y - Frame::cy) * z * Frame::invfy,
	                                z);
	                            const Eigen::Vector3f Xcur = Trel * Xprev;
	                            if(!(Xcur.z() > 1e-6f))
	                                continue;

	                            const Eigen::Vector2f uv_hat = mpCamera->project(Xcur);
	                            if(!std::isfinite(uv_hat.x()) || !std::isfinite(uv_hat.y()))
	                                continue;
	                            if(uv_hat.x() < 0.0f || uv_hat.y() < 0.0f || uv_hat.x() >= static_cast<float>(mImGray.cols) ||
	                               uv_hat.y() >= static_cast<float>(mImGray.rows))
	                                continue;

	                            prev_pts.push_back(uv_prev);
	                            const cv::Point2f uv_hat_cv(uv_hat.x(), uv_hat.y());
	                            hat_pts.push_back(uv_hat_cv);
	                            next_pts.push_back(uv_hat_cv); // initial guess
	                        }

	                        if(prev_pts.empty())
	                            return 0;

	                        std::vector<uchar> status;
	                        std::vector<float> err;
	                        const cv::Size win_size(21, 21);
	                        const int max_level = 3;
	                        const cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
	                        const int flags = cv::OPTFLOW_USE_INITIAL_FLOW;
	                        cv::calcOpticalFlowPyrLK(
	                            mImGrayPrev,
	                            mImGray,
	                            prev_pts,
	                            next_pts,
	                            status,
	                            err,
	                            win_size,
	                            max_level,
	                            termcrit,
	                            flags,
	                            1e-4);

	                        int kept = 0;
	                        const size_t n = std::min(prev_pts.size(), std::min(next_pts.size(), hat_pts.size()));
	                        for(size_t i = 0; i < n; ++i)
	                        {
	                            if(i >= status.size() || status[i] == 0)
	                                continue;
	                            const cv::Point2f uv_lk = next_pts[i];
	                            const cv::Point2f uv_hat = hat_pts[i];
	                            const float dx = uv_lk.x - uv_hat.x;
	                            const float dy = uv_lk.y - uv_hat.y;
	                            const float e = std::sqrt(dx * dx + dy * dy);
	                            if(!std::isfinite(e))
	                                continue;

	                            int gid = GetDIFGlobalTrackIdAt(uv_lk);
	                            if(gid < 0)
	                                gid = gid_at_prev(prev_pts[i]);
	                            if(gid >= 0)
	                                flow_inst[gid].push_back(e);
	                            else
	                                flow_bg.push_back(e);
	                            kept++;
	                        }
	                        return kept;
	                    };

	                    int kept = 0;
	                    if(mDIFStateCfg.flow_mode == 1)
	                    {
	                        kept = compute_flow_lk();
	                        if(kept < 5)
	                        {
	                            // Fallback to descriptor matching when LK cannot produce enough valid tracks.
	                            flow_bg.clear();
	                            flow_inst.clear();
	                            kept = compute_flow_desc();
	                        }
	                    }
	                    else
	                    {
	                        kept = compute_flow_desc();
	                    }
	                }

                std::unordered_map<int, std::vector<float>> inst_errors;
                std::unordered_map<int, DIFTrackState> prev_states;
                {
                    auto& tracks_mut = mpDIFInstanceTracker->GetTracksMutable();
                    prev_states.reserve(tracks_mut.size());
                    for(const auto& kv : tracks_mut)
                        prev_states.emplace(kv.first, kv.second.state_hat);
                }
                const float r_bg = mpDIFStateEstimator->UpdateFromFrame(
                    mCurrentFrame,
                    mCurrentFrame.mpCamera,
                    mpDIFInstanceTracker->GetTracksMutable(),
                    inst_errors,
                    label_map,
                    local_to_global,
                    flow_bg.empty() ? nullptr : &flow_bg,
                    flow_inst.empty() ? nullptr : &flow_inst);
                UpdateDIFSuppressedTrackIds();
                UpdateDIFForbiddenForMapTrackIds();
                UpdateDIFTrackLockInfoSnapshot();

                // DIF-SLAM v2.0 rollback trigger: when a track enters D (nonD->D), rollback its historical sparse/dense contributions.
                std::vector<int> entered_d;
                {
                    const auto& tracks_now = mpDIFInstanceTracker->GetTracks();
                    entered_d.reserve(8);
                    for(const auto& kv : tracks_now)
                    {
                        const int gid = kv.first;
                        if(gid < 0)
                            continue;
                        const DIFTrackState cur = kv.second.state_hat;
                        if(cur != DIFTrackState::D)
                            continue;
                        const auto it = prev_states.find(gid);
                        const DIFTrackState prev = (it != prev_states.end()) ? it->second : DIFTrackState::S;
                        if(prev != DIFTrackState::D)
                            entered_d.push_back(gid);
                    }
                }
                if(!entered_d.empty())
                {
                    if(mpLocalMapper)
                        mpLocalMapper->EnqueueDIFPruneTrackIds(entered_d);
                    if(mpDenseMapping && mDenseCfg.enable)
                    {
                        for(const int gid : entered_d)
                            mpDenseMapping->EnqueueTrackEnterD(gid, fid_cur, mCurrentFrame.mTimeStamp);
                    }
                    std::cout << "[DIF-ROLLBACK] frame_id=" << fid_cur
                              << " enter_D_tracks=" << entered_d.size()
                              << std::endl;
                }
                mDIFPrevDynMask = mDIFDynMaskLast;
                mDIFPrevDynMaskFrameId = mDIFLastDynMaskFrameId;
                mDIFDynMaskLast = BuildDIFDynamicMaskFromTracks(mImGray.rows, mImGray.cols, mCurrentFrame.mTimeStamp);
                mDIFLastDynMaskFrameId = static_cast<int>(mCurrentFrame.mnId);
                mDIFLastDynMaskTimestamp = mCurrentFrame.mTimeStamp;

                // Track states/masks have been updated for this frame: invalidate per-frame predicted masks so that
                // subsequent uses (e.g., second-pass) rebuild with the latest states (P-002).
                InvalidateDIFPredMasks();

                // Optional enhancement: second-pass PoseOptimization using current M_t^{dyn}.
                // This follows the optimized DIF spec (run after first-pass pose + state estimation).
                MaybeDIFSecondPassPoseOptimization();

                // Store pose-quality metric for this frame, so late-arriving segmentations can gate 3D update (Module B 8.2).
                for(auto& e : mDIFFrameCache)
                {
                    if(e.frame_id == fid_cur)
                    {
                        e.has_r_bg = std::isfinite(r_bg);
                        e.r_bg = r_bg;
                        break;
                    }
                }

                // Lightweight observability on segmentation-update frames (Module C key metrics).
                // DIF-SLAM v2.0: two-state HMM {S, D} + map_lock (lock period, displayed as S*).
                // - S: state_hat==S && map_lock==false (stable static)
                // - S*: state_hat==S && map_lock==true (lock period: participate but downweighted, rollback on nonD->D)
                // - D: state_hat==D (dynamic)
                if(mDIFLocalToGlobalFrameId == fid_cur)
                {
                    int nS = 0, nSLock = 0, nD = 0;
                    int nStaticSupp = 0;
                    const int k_static = std::max(1, mDIFStateCfg.static_K);
                    for(const auto& kv : mpDIFInstanceTracker->GetTracks())
                    {
                        const DIFTrack& tr = kv.second;
                        if(tr.state_hat == DIFTrackState::D)
                        {
                            nD++;
                        }
                        else if(tr.map_lock)
                        {
                            nSLock++;  // S* (lock period)
                        }
                        else
                        {
                            nS++;
                        }
                        if(mDIFStateCfg.static_enable && (tr.static_err_count >= k_static || tr.static_flow_count >= k_static))
                            nStaticSupp++;
                    }
                    const int dyn_area = (!mDIFDynMaskLast.empty() && mDIFDynMaskLast.type() == CV_8U)
                                             ? cv::countNonZero(mDIFDynMaskLast)
                                             : 0;
                    std::cout << "[DIF-C] updated at frame_id=" << fid_cur
                              << " r_bg=" << (std::isfinite(r_bg) ? r_bg : -1.0f)
                              << " tracks=" << mpDIFInstanceTracker->GetTracks().size()
                              << " S=" << nS << " S*=" << nSLock << " D=" << nD
                              << " static_supp=" << nStaticSupp
                              << " dyn_area=" << dyn_area
                              << std::endl;
                }
            }
        }

        SaveDIFBinaryMaskForCurrentFrame();
        SaveFlowVisForCurrentFrame();

        // Update drawer
        mpFrameDrawer->Update(this);
        if(mCurrentFrame.isSet())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        if(bOK || mState==RECENTLY_LOST)
        {
            // Update motion model
            if(mLastFrame.isSet() && mCurrentFrame.isSet())
            {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                mVelocity = mCurrentFrame.GetPose() * LastTwc;
                mbVelocity = true;
            }
            else {
                mbVelocity = false;
            }

            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = NeedNewKeyFrame();

            // Check if we need to insert a new keyframe
            // if(bNeedKF && bOK)
            if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST &&
                                   (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                CreateNewKeyFrame();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(pCurrentMap->KeyFramesInMap()<=10)
            {
                mpSystem->ResetActiveMap();
                return;
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }

            CreateMapInAtlas();

            return;
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }




    if(mState==OK || mState==RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(mCurrentFrame.isSet())
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState==LOST);
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);
        }

    }

#ifdef REGISTER_LOOP
    if (Stop()) {

        // Safe area to stop
        while(isStopped())
        {
            usleep(3000);
        }
    }
#endif
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA).norm()<0.5)
            {
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        else
            mCurrentFrame.SetPose(Sophus::SE3f());

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        const cv::Mat* forbid_mask = nullptr;
        int dif_init_skipped_forbid = 0;
        int dif_init_skipped_track = 0;
        if(mDIFSegCfg.enable && mDIFFilterEnable && mDIFStateCfg.enable && mpDIFInstanceTracker)
        {
            pKFini->SetDIFForbidMask(GetDIFForbidMaskPredForCurrentFrame());
            forbid_mask = &pKFini->GetDIFForbidMask();
        }

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                const cv::Point2f pt = mCurrentFrame.mvKeys[i].pt;
                if(mDIFSegCfg.enable && mDIFFilterEnable)
                {
                    bool in_forbid = (forbid_mask && !forbid_mask->empty() && IsInDIFMask(*forbid_mask, pt));
                    if(in_forbid)
                    {
                        dif_init_skipped_forbid++;
                        continue;
                    }
                }
                const int gid = GetDIFGlobalTrackIdAt(pt);
                if(mDIFSegCfg.enable && mDIFFilterEnable &&
                   (IsDIFTrackSuppressed(gid) || IsDIFTrackForbiddenForMap(gid)))
                {
                    dif_init_skipped_track++;
                    continue;
                }
                float z = mCurrentFrame.mvDepth[i];
                if(z>0)
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereo(i, x3D);
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->mnInstanceId = gid;
                    pNewMP->mnDIFCreatedFrameId = static_cast<int>(mCurrentFrame.mnId);
                    if(gid >= 0 && mpDIFInstanceTracker)
                    {
                        const auto& tracks = mpDIFInstanceTracker->GetTracks();
                        const auto it = tracks.find(gid);
                        if(it != tracks.end())
                        {
                            pNewMP->mbDIFMapLockAtCreate = it->second.map_lock;
                            pNewMP->mfDIFWLockAtCreate = it->second.w_lock;
                        }
                    }
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    const cv::Point2f pt = mCurrentFrame.mvKeys[i].pt;
                    if(mDIFSegCfg.enable && mDIFFilterEnable)
                    {
                        bool in_forbid = (forbid_mask && !forbid_mask->empty() && IsInDIFMask(*forbid_mask, pt));
                        if(in_forbid)
                        {
                            dif_init_skipped_forbid++;
                            continue;
                        }
                    }
                    const int gid = GetDIFGlobalTrackIdAt(pt);
                    if(mDIFSegCfg.enable && mDIFFilterEnable &&
                       (IsDIFTrackSuppressed(gid) || IsDIFTrackForbiddenForMap(gid)))
                    {
                        dif_init_skipped_track++;
                        continue;
                    }
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->mnInstanceId = gid;
                    pNewMP->mnDIFCreatedFrameId = static_cast<int>(mCurrentFrame.mnId);
                    if(gid >= 0 && mpDIFInstanceTracker)
                    {
                        const auto& tracks = mpDIFInstanceTracker->GetTracks();
                        const auto it = tracks.find(gid);
                        if(it != tracks.end())
                        {
                            pNewMP->mbDIFMapLockAtCreate = it->second.map_lock;
                            pNewMP->mfDIFWLockAtCreate = it->second.w_lock;
                        }
                    }

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        if(mDIFSegCfg.enable && mDIFFilterEnable && (dif_init_skipped_forbid + dif_init_skipped_track) > 0)
        {
            std::cout << "[DIF-INIT] frame_id=" << static_cast<int>(mCurrentFrame.mnId)
                      << " kf=" << pKFini->mnId
                      << " skipped_forbid=" << dif_init_skipped_forbid
                      << " skipped_track=" << dif_init_skipped_track
                      << std::endl;
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        //cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        //mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        mState=OK;
    }
}


void Tracking::MonocularInitialization()
{

    if(!mbReadyToInitializate)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {

            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            if (mSensor == System::IMU_MONOCULAR)
            {
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }

            mbReadyToInitializate = true;

            return;
        }
    }
    else
    {
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            mbReadyToInitializate = false;

            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            mbReadyToInitializate = false;
            return;
        }

        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvIniMatches,Tcw,mvIniP3D,vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(Sophus::SE3f());
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}



void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }


    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap();
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }


    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    //mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;

    initID = pKFcur->mnId;
}


void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame.mnId+1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
    //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mbReadyToInitializate = false;
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
    {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    if(mDIFSegCfg.enable && mDIFFilterEnable)
        ApplyDIFMaskToMatchedMapPoints(mCurrentFrame);

    //mCurrentFrame.PrintPointDistribution();


    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        //if(i >= mCurrentFrame.Nleft) break;
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    Sophus::SE3f Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    const int Nfeat = mLastFrame.Nleft == -1? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for(int i=0; i<Nfeat;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
            bCreateNew = true;

        if(bCreateNew)
        {
            Eigen::Vector3f x3D;

            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else{
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);
            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;

    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
        return true;
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    }




    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;

    if(mSensor==System::STEREO)
        th=7;
    else
        th=15;

    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
        return false;
    }

    if(mDIFSegCfg.enable && mDIFFilterEnable)
        ApplyDIFMaskToMatchedMapPoints(mCurrentFrame);

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;

    UpdateLocalMap();
    SearchLocalPoints();

    if(mDIFSegCfg.enable && mDIFFilterEnable)
        ApplyDIFMaskToMatchedMapPoints(mCurrentFrame);

    // TOO check outliers before PO
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    int inliers;
    if (!mpAtlas->isImuInitialized())
        Optimizer::PoseOptimization(&mCurrentFrame);
    else
    {
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
        {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else
        {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            if(!mbMapUpdated) //  && (mnMatchesInliers>30))
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers;
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST))
        return true;


    if (mSensor == System::IMU_MONOCULAR)
    {
        if((mnMatchesInliers<15 && mpAtlas->isImuInitialized())||(mnMatchesInliers<50 && !mpAtlas->isImuInitialized()))
        {
            return false;
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else
    {
        if(mnMatchesInliers<30)
            return false;
        else
            return true;
    }
}

bool Tracking::NeedNewKeyFrame()
{
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
        //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && mSensor!=System::IMU_RGBD && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
            {
                //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                return false;
            }
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    const cv::Mat* forbid_mask = nullptr;
    int dif_kf_skipped_forbid = 0;
    int dif_kf_skipped_track = 0;
    if(mDIFSegCfg.enable && mDIFFilterEnable && mDIFStateCfg.enable && mpDIFInstanceTracker)
    {
        pKF->SetDIFForbidMask(GetDIFForbidMaskPredForCurrentFrame());
        forbid_mask = &pKF->GetDIFForbidMask();
    }

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;
    MaybeEnqueueDenseMappingFromKeyFrame(pKF);

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    const cv::Point2f pt = mCurrentFrame.mvKeys[i].pt;
                    if(mDIFSegCfg.enable && mDIFFilterEnable)
                    {
                        bool in_forbid = (forbid_mask && !forbid_mask->empty() && IsInDIFMask(*forbid_mask, pt));
                        if(in_forbid)
                        {
                            dif_kf_skipped_forbid++;
                            continue;
                        }
                    }
                    const int gid = GetDIFGlobalTrackIdAt(pt);
                    if(mDIFSegCfg.enable && mDIFFilterEnable &&
                       (IsDIFTrackSuppressed(gid) || IsDIFTrackForbiddenForMap(gid)))
                    {
                        dif_kf_skipped_track++;
                        continue;
                    }
                    Eigen::Vector3f x3D;

                    if(mCurrentFrame.Nleft == -1){
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->mnInstanceId = gid;
                    pNewMP->mnDIFCreatedFrameId = static_cast<int>(mCurrentFrame.mnId);
                    if(gid >= 0 && mpDIFInstanceTracker)
                    {
                        const auto& tracks = mpDIFInstanceTracker->GetTracks();
                        const auto it = tracks.find(gid);
                        if(it != tracks.end())
                        {
                            pNewMP->mbDIFMapLockAtCreate = it->second.map_lock;
                            pNewMP->mfDIFWLockAtCreate = it->second.w_lock;
                        }
                    }
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }
            //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
    }

    if(mDIFSegCfg.enable && mDIFFilterEnable && (dif_kf_skipped_forbid + dif_kf_skipped_track) > 0)
    {
        std::cout << "[DIF-KF] frame_id=" << static_cast<int>(mCurrentFrame.mnId)
                  << " kf=" << pKF->mnId
                  << " skipped_forbid=" << dif_kf_skipped_forbid
                  << " skipped_track=" << dif_kf_skipped_track
                  << std::endl;
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
        if(pMP->mbTrackInView)
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            th=3;
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=6;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15; // 15

        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {

            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }


    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 80
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&mvpLocalKeyFrames.size()<80)
    {
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(bTcw)
            {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw);
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();
    ResetDIFRuntimeState(true);

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();
    ResetDIFRuntimeState(false);


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

} //namespace ORB_SLAM
