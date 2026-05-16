#ifndef ORB_SLAM3_DIF_SEGMENTATION_WORKER_H
#define ORB_SLAM3_DIF_SEGMENTATION_WORKER_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>

namespace ORB_SLAM3 {

struct SegmentationConfig
{
    bool enable = false;
    std::string python = "python3";
    std::string server_script = "scripts/dif/fastsam_server.py";
    std::string weights = "weights/FastSAM_X.pt";
    std::string device = "cuda:0";

    int every_n_frames = 3;
    int jpeg_quality = 80;
    // Per-request IPC timeout (read/write). If exceeded, worker restarts the python process.
    int req_timeout_ms = 5000;
    // Backoff before restarting python process after a failure.
    int restart_backoff_ms = 200;

    int imgsz = 1024;
    float conf = 0.4f;
    float iou = 0.9f;
    bool retina_masks = true;
    // Save raw FastSAM Everything output visualization (unprocessed masks overlay).
    // Output: Output/everything_mask/frame_XXXXXX.png (saved on segmentation frames).
    bool everything_raw_vis_enable = true;
    // Save Module-A post-processing visualization (final label_map overlay; no text).
    // Output: Output/post_processing/frame_XXXXXX.png (saved on segmentation frames).
    bool post_processing_vis_enable = true;

    int area_min = 600;
    float area_max_ratio = 0.65f;
    // Drop masks that touch too many image borders (>=3 usually indicates background-like regions).
    // Set <=0 to disable.
    int border_touch_min_sides = 3;
    int morph_kernel = 3;
    float iou_nms = 0.9f;
    int max_masks = 50;

    // Per-mask connected-components cleanup (0 disables).
    int cc_min_area = 0;
    // Overlap-resolution assignment thresholds:
    // Only accept a new label if it contributes enough free pixels to label_map.
    int min_assign_pixels = 0;
    float min_assign_ratio = 0.0f;
    // Global label_map cleanup: remove/merge tiny islands (0 disables).
    int island_min_area = 0;
};

struct SegmentationResult
{
    bool ok = false;
    int frame_id = -1;
    double timestamp = -1.0;
    double elapsed_ms = 0.0;
    std::string error;
    cv::Mat label_map; // CV_16S, value range [-1, N-1]
};

class SegmentationWorker
{
public:
    explicit SegmentationWorker(const SegmentationConfig& cfg);
    ~SegmentationWorker();

    bool Start();
    void Stop();

    void SubmitFrame(int frame_id, double timestamp, const cv::Mat& image_bgr_or_rgb, bool is_rgb);
    bool TryGetLatest(SegmentationResult& out);
    bool WaitForFrameResult(int frame_id, int timeout_ms, SegmentationResult& out);

private:
    void ThreadMain();
    bool EnsureProcessStarted();
    void StopProcess(const char* reason, bool backoff);

    bool SendSegmentRequest(int frame_id, double timestamp, const std::vector<uchar>& jpeg_bytes, bool is_rgb);
    bool ReceiveSegmentResponse(SegmentationResult& out, std::string& out_err);

private:
    SegmentationConfig mCfg;

    std::atomic<bool> mbStop{false};
    std::thread mThread;

    std::mutex mMutexQueue;
    std::condition_variable mCvQueue;
    bool mHasPending = false;
    int mPendingFrameId = -1;
    double mPendingTimestamp = -1.0;
    bool mPendingIsRgb = false;
    std::vector<uchar> mPendingJpeg;

    std::mutex mMutexLatest;
    std::condition_variable mCvLatest;
    SegmentationResult mLatest;

    int mChildPid = -1;
    int mFdToChild = -1;
    int mFdFromChild = -1;
    int64_t mLastProcessStopMs = 0;
    bool mLastProcessStopWasFailure = false;
};

} // namespace ORB_SLAM3

#endif
