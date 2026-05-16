#include "DIF/SegmentationWorker.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <thread>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>

namespace ORB_SLAM3 {
namespace {

int64_t NowMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

bool WaitFd(int fd, short events, int timeout_ms)
{
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = events;
    pfd.revents = 0;
    const int r = ::poll(&pfd, 1, timeout_ms);
    if(r <= 0)
        return false;
    if(pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
        return false;
    return (pfd.revents & events) != 0;
}

bool WriteAllTimeout(int fd, const void* data, size_t size, int timeout_ms)
{
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t n = 0;
    const int64_t deadline = NowMs() + std::max(0, timeout_ms);
    while(n < size)
    {
        if(timeout_ms > 0)
        {
            const int64_t now = NowMs();
            const int remaining = static_cast<int>(std::max<int64_t>(0, deadline - now));
            if(!WaitFd(fd, POLLOUT, remaining))
                return false;
        }

        const ssize_t w = ::write(fd, p + n, size - n);
        if(w < 0)
        {
            if(errno == EINTR) continue;
            if(errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return false;
        }
        n += static_cast<size_t>(w);
    }
    return true;
}

bool ReadAllTimeout(int fd, void* data, size_t size, int timeout_ms)
{
    uint8_t* p = static_cast<uint8_t*>(data);
    size_t n = 0;
    const int64_t deadline = NowMs() + std::max(0, timeout_ms);
    while(n < size)
    {
        if(timeout_ms > 0)
        {
            const int64_t now = NowMs();
            const int remaining = static_cast<int>(std::max<int64_t>(0, deadline - now));
            if(!WaitFd(fd, POLLIN, remaining))
                return false;
        }

        const ssize_t r = ::read(fd, p + n, size - n);
        if(r == 0) return false;
        if(r < 0)
        {
            if(errno == EINTR) continue;
            if(errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return false;
        }
        n += static_cast<size_t>(r);
    }
    return true;
}

uint32_t ToBE32(uint32_t x)
{
    return ((x & 0x000000FFu) << 24) | ((x & 0x0000FF00u) << 8) | ((x & 0x00FF0000u) >> 8) | ((x & 0xFF000000u) >> 24);
}

uint32_t FromBE32(uint32_t x) { return ToBE32(x); }

bool WriteMessageTimeout(int fd, const std::string& header_json, const std::vector<uint8_t>& payload, int timeout_ms)
{
    const uint32_t header_len_be = ToBE32(static_cast<uint32_t>(header_json.size()));
    if(!WriteAllTimeout(fd, &header_len_be, sizeof(header_len_be), timeout_ms)) return false;
    if(!WriteAllTimeout(fd, header_json.data(), header_json.size(), timeout_ms)) return false;
    if(!payload.empty())
    {
        if(!WriteAllTimeout(fd, payload.data(), payload.size(), timeout_ms)) return false;
    }
    return true;
}

bool ReadMessageTimeout(int fd, std::string& header_json, std::vector<uint8_t>& payload, int timeout_ms)
{
    uint32_t header_len_be = 0;
    if(!ReadAllTimeout(fd, &header_len_be, sizeof(header_len_be), timeout_ms)) return false;
    const uint32_t header_len = FromBE32(header_len_be);
    if(header_len == 0 || header_len > (1u << 20)) return false; // sanity: header <= 1MB
    header_json.assign(header_len, '\0');
    if(!ReadAllTimeout(fd, &header_json[0], header_len, timeout_ms)) return false;

    const std::string key = "\"payload_len\":";
    const size_t pos = header_json.find(key);
    size_t payload_len = 0;
    if(pos != std::string::npos)
    {
        size_t i = pos + key.size();
        while(i < header_json.size() && (header_json[i] == ' ')) i++;
        size_t j = i;
        while(j < header_json.size() && (header_json[j] >= '0' && header_json[j] <= '9')) j++;
        if(j > i)
        {
            payload_len = static_cast<size_t>(std::strtoull(header_json.substr(i, j - i).c_str(), nullptr, 10));
        }
    }

    payload.assign(payload_len, 0);
    if(payload_len > 0)
    {
        if(payload_len > (64u << 20)) return false; // sanity: payload <= 64MB
        if(!ReadAllTimeout(fd, payload.data(), payload_len, timeout_ms)) return false;
    }
    return true;
}

} // namespace

SegmentationWorker::SegmentationWorker(const SegmentationConfig& cfg) : mCfg(cfg) {}

SegmentationWorker::~SegmentationWorker() { Stop(); }

bool SegmentationWorker::Start()
{
    if(mThread.joinable()) return true;
    mbStop = false;
    mThread = std::thread(&SegmentationWorker::ThreadMain, this);
    return true;
}

void SegmentationWorker::Stop()
{
    mbStop = true;
    mCvQueue.notify_all();
    if(mThread.joinable()) mThread.join();
    StopProcess("stop", false);
}

void SegmentationWorker::SubmitFrame(int frame_id, double timestamp, const cv::Mat& image_bgr_or_rgb, bool is_rgb)
{
    if(!mCfg.enable) return;
    if(image_bgr_or_rgb.empty()) return;

    std::lock_guard<std::mutex> lk(mMutexQueue);
    mHasPending = true;
    mPendingFrameId = frame_id;
    mPendingTimestamp = timestamp;
    mPendingIsRgb = is_rgb;
    mPendingImage = image_bgr_or_rgb.clone();
    mCvQueue.notify_one();
}

bool SegmentationWorker::TryGetLatest(SegmentationResult& out)
{
    std::lock_guard<std::mutex> lk(mMutexLatest);
    if(mLatest.frame_id < 0) return false;
    out = mLatest;
    return true;
}

bool SegmentationWorker::WaitForFrameResult(int frame_id, int timeout_ms, SegmentationResult& out)
{
    std::unique_lock<std::mutex> lk(mMutexLatest);
    const auto ready = [&]() {
        return mbStop || (mLatest.frame_id >= frame_id && mLatest.frame_id >= 0);
    };
    if(!ready())
    {
        if(timeout_ms <= 0)
            return false;
        mCvLatest.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
    }
    if(!ready() || mLatest.frame_id < frame_id)
        return false;
    out = mLatest;
    return true;
}

void SegmentationWorker::ThreadMain()
{
    while(!mbStop)
    {
        int frame_id = -1;
        double timestamp = -1.0;
        bool is_rgb = false;
        cv::Mat image;

        {
            std::unique_lock<std::mutex> lk(mMutexQueue);
            mCvQueue.wait(lk, [&]{ return mbStop || mHasPending; });
            if(mbStop) break;
            frame_id = mPendingFrameId;
            timestamp = mPendingTimestamp;
            is_rgb = mPendingIsRgb;
            image = std::move(mPendingImage);
            mHasPending = false;
        }

        if(frame_id < 0 || image.empty())
            continue;

        std::vector<int> params;
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(std::max(1, std::min(100, mCfg.jpeg_quality)));

        std::vector<uchar> jpeg;
        if(!cv::imencode(".jpg", image, jpeg, params))
            continue;

        if(!EnsureProcessStarted())
        {
            SegmentationResult res;
            res.ok = false;
            res.frame_id = frame_id;
            res.timestamp = timestamp;
            res.error = "start_process_failed";
            {
                std::lock_guard<std::mutex> lk(mMutexLatest);
                mLatest = std::move(res);
            }
            mCvLatest.notify_all();
            continue;
        }

        if(!SendSegmentRequest(frame_id, timestamp, jpeg, is_rgb))
        {
            SegmentationResult res;
            res.ok = false;
            res.frame_id = frame_id;
            res.timestamp = timestamp;
            res.error = "send_request_failed";
            {
                std::lock_guard<std::mutex> lk(mMutexLatest);
                mLatest = std::move(res);
            }
            mCvLatest.notify_all();
            StopProcess("send_request_failed", true);
            continue;
        }

        SegmentationResult res;
        std::string recv_err;
        if(!ReceiveSegmentResponse(res, recv_err))
        {
            SegmentationResult fail;
            fail.ok = false;
            fail.frame_id = frame_id;
            fail.timestamp = timestamp;
            fail.error = !recv_err.empty() ? recv_err : "receive_response_failed";
            {
                std::lock_guard<std::mutex> lk(mMutexLatest);
                mLatest = std::move(fail);
            }
            mCvLatest.notify_all();
            StopProcess("receive_response_failed", true);
            continue;
        }

        if(res.ok && res.frame_id != frame_id)
        {
            SegmentationResult fail;
            fail.ok = false;
            fail.frame_id = frame_id;
            fail.timestamp = timestamp;
            fail.error = "frame_id_mismatch";
            {
                std::lock_guard<std::mutex> lk(mMutexLatest);
                mLatest = std::move(fail);
            }
            mCvLatest.notify_all();
            StopProcess("frame_id_mismatch", true);
            continue;
        }

        std::lock_guard<std::mutex> lk(mMutexLatest);
        mLatest = std::move(res);
        mCvLatest.notify_all();
    }

    if(mFdToChild >= 0 && mFdFromChild >= 0)
    {
        const std::string header = "{\"type\":\"shutdown\"}";
        WriteMessageTimeout(mFdToChild, header, {}, 200);
    }
}

bool SegmentationWorker::EnsureProcessStarted()
{
    if(mChildPid > 0 && mFdToChild >= 0 && mFdFromChild >= 0)
        return true;

    if(mCfg.restart_backoff_ms > 0 && mLastProcessStopMs > 0 && mLastProcessStopWasFailure)
    {
        const int64_t now = NowMs();
        const int64_t elapsed = now - mLastProcessStopMs;
        if(elapsed < mCfg.restart_backoff_ms)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(mCfg.restart_backoff_ms - elapsed)));
        }
    }

    int to_child[2] = {-1, -1};
    int from_child[2] = {-1, -1};
    if(::pipe(to_child) != 0) return false;
    if(::pipe(from_child) != 0)
    {
        ::close(to_child[0]);
        ::close(to_child[1]);
        return false;
    }

    const pid_t pid = ::fork();
    if(pid < 0)
    {
        ::close(to_child[0]); ::close(to_child[1]);
        ::close(from_child[0]); ::close(from_child[1]);
        return false;
    }

    if(pid == 0)
    {
        ::dup2(to_child[0], STDIN_FILENO);
        ::dup2(from_child[1], STDOUT_FILENO);

        ::close(to_child[0]); ::close(to_child[1]);
        ::close(from_child[0]); ::close(from_child[1]);

        ::setenv("PYTHONUNBUFFERED", "1", 1);

        std::vector<std::string> args;
        args.push_back(mCfg.python);
        args.push_back("-u");
        args.push_back(mCfg.server_script);
        args.push_back("--weights");
        args.push_back(mCfg.weights);
        args.push_back("--device");
        args.push_back(mCfg.device);
        args.push_back("--imgsz");
        args.push_back(std::to_string(mCfg.imgsz));
        args.push_back("--conf");
        args.push_back(std::to_string(mCfg.conf));
        args.push_back("--iou");
        args.push_back(std::to_string(mCfg.iou));
        if(mCfg.retina_masks) args.push_back("--retina-masks");
        args.push_back("--half");
        args.push_back(mCfg.half ? "1" : "0");
        args.push_back("--deterministic");
        args.push_back(mCfg.deterministic ? "1" : "0");
        args.push_back("--save-everything-vis");
        args.push_back(mCfg.everything_raw_vis_enable ? "1" : "0");
        args.push_back("--save-post-vis");
        args.push_back(mCfg.post_processing_vis_enable ? "1" : "0");
        args.push_back("--area-min");
        args.push_back(std::to_string(mCfg.area_min));
        args.push_back("--area-max-ratio");
        args.push_back(std::to_string(mCfg.area_max_ratio));
        if(mCfg.border_touch_min_sides > 0)
        {
            args.push_back("--border-touch-min-sides");
            args.push_back(std::to_string(mCfg.border_touch_min_sides));
        }
        args.push_back("--morph-kernel");
        args.push_back(std::to_string(mCfg.morph_kernel));
        args.push_back("--iou-nms");
        args.push_back(std::to_string(mCfg.iou_nms));
        args.push_back("--max-masks");
        args.push_back(std::to_string(mCfg.max_masks));
        if(mCfg.cc_min_area > 0)
        {
            args.push_back("--cc-min-area");
            args.push_back(std::to_string(mCfg.cc_min_area));
        }
        if(mCfg.min_assign_pixels > 0)
        {
            args.push_back("--min-assign-pixels");
            args.push_back(std::to_string(mCfg.min_assign_pixels));
        }
        if(mCfg.min_assign_ratio > 0.0f)
        {
            args.push_back("--min-assign-ratio");
            args.push_back(std::to_string(mCfg.min_assign_ratio));
        }
        if(mCfg.island_min_area > 0)
        {
            args.push_back("--island-min-area");
            args.push_back(std::to_string(mCfg.island_min_area));
        }

        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for(std::string& s : args) argv.push_back(const_cast<char*>(s.c_str()));
        argv.push_back(nullptr);

        ::execvp(argv[0], argv.data());
        ::_exit(127);
    }

    ::close(to_child[0]);
    ::close(from_child[1]);

    mChildPid = static_cast<int>(pid);
    mFdToChild = to_child[1];
    mFdFromChild = from_child[0];

    ::fcntl(mFdToChild, F_SETFD, FD_CLOEXEC);
    ::fcntl(mFdFromChild, F_SETFD, FD_CLOEXEC);
    ::fcntl(mFdToChild, F_SETFL, ::fcntl(mFdToChild, F_GETFL, 0) | O_NONBLOCK);
    ::fcntl(mFdFromChild, F_SETFL, ::fcntl(mFdFromChild, F_GETFL, 0) | O_NONBLOCK);

    std::cout << "[DIF-A] started pid=" << mChildPid << std::endl;
    return true;
}

void SegmentationWorker::StopProcess(const char* reason, bool backoff)
{
    if(mFdToChild >= 0) { ::close(mFdToChild); mFdToChild = -1; }
    if(mFdFromChild >= 0) { ::close(mFdFromChild); mFdFromChild = -1; }

    if(mChildPid > 0)
    {
        const int pid = mChildPid;
        mChildPid = -1;

        int status = 0;
        const pid_t r0 = ::waitpid(pid, &status, WNOHANG);
        if(r0 == 0)
        {
            ::kill(pid, SIGTERM);
            for(int i = 0; i < 20; ++i)
            {
                const pid_t r = ::waitpid(pid, &status, WNOHANG);
                if(r == pid)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            const pid_t r = ::waitpid(pid, &status, WNOHANG);
            if(r == 0)
            {
                ::kill(pid, SIGKILL);
                ::waitpid(pid, &status, 0);
            }
        }

        mLastProcessStopMs = NowMs();
        mLastProcessStopWasFailure = backoff;
        if(reason && reason[0] != '\0')
            std::cout << "[DIF-A] stopped pid=" << pid << " reason=" << reason << std::endl;
    }
}

bool SegmentationWorker::SendSegmentRequest(int frame_id, double timestamp, const std::vector<uchar>& jpeg_bytes, bool is_rgb)
{
    std::string header = "{";
    header += "\"type\":\"segment\",";
    header += "\"frame_id\":" + std::to_string(frame_id) + ",";
    header += "\"timestamp\":" + std::to_string(timestamp) + ",";
    header += "\"is_rgb\":" + std::to_string(is_rgb ? 1 : 0) + ",";
    header += "\"payload_len\":" + std::to_string(jpeg_bytes.size());
    header += "}";

    std::vector<uint8_t> payload(jpeg_bytes.begin(), jpeg_bytes.end());
    return WriteMessageTimeout(mFdToChild, header, payload, std::max(0, mCfg.req_timeout_ms));
}

bool SegmentationWorker::ReceiveSegmentResponse(SegmentationResult& out, std::string& out_err)
{
    std::string header;
    std::vector<uint8_t> payload;
    out_err.clear();
    if(!ReadMessageTimeout(mFdFromChild, header, payload, std::max(0, mCfg.req_timeout_ms)))
    {
        out_err = "receive_timeout_or_io_error";
        return false;
    }

    out = SegmentationResult();

    auto find_bool = [&](const std::string& key, bool def) -> bool {
        const std::string k = "\"" + key + "\":";
        const size_t p = header.find(k);
        if(p == std::string::npos) return def;
        size_t v = p + k.size();
        while(v < header.size() && header[v] == ' ') v++;
        if(header.compare(v, 4, "true") == 0) return true;
        if(header.compare(v, 5, "false") == 0) return false;
        if(v < header.size() && (header[v] == '1')) return true;
        if(v < header.size() && (header[v] == '0')) return false;
        return def;
    };

    auto find_int = [&](const std::string& key, int def) -> int {
        const std::string k = "\"" + key + "\":";
        const size_t p = header.find(k);
        if(p == std::string::npos) return def;
        size_t i = p + k.size();
        while(i < header.size() && header[i] == ' ') i++;
        size_t j = i;
        if(j < header.size() && header[j] == '-') j++;
        while(j < header.size() && (header[j] >= '0' && header[j] <= '9')) j++;
        if(j <= i) return def;
        return std::atoi(header.substr(i, j - i).c_str());
    };

    auto find_double = [&](const std::string& key, double def) -> double {
        const std::string k = "\"" + key + "\":";
        const size_t p = header.find(k);
        if(p == std::string::npos) return def;
        size_t i = p + k.size();
        while(i < header.size() && header[i] == ' ') i++;
        size_t j = i;
        if(j < header.size() && header[j] == '-') j++;
        while(j < header.size() && ((header[j] >= '0' && header[j] <= '9') || header[j] == '.')) j++;
        if(j <= i) return def;
        return std::atof(header.substr(i, j - i).c_str());
    };

    auto find_string = [&](const std::string& key) -> std::string {
        const std::string k = "\"" + key + "\":";
        const size_t p = header.find(k);
        if(p == std::string::npos) return "";
        size_t i = p + k.size();
        while(i < header.size() && header[i] == ' ') i++;
        if(i >= header.size() || header[i] != '"') return "";
        i++;
        size_t j = i;
        bool esc = false;
        while(j < header.size())
        {
            const char c = header[j];
            if(esc)
            {
                esc = false;
                j++;
                continue;
            }
            if(c == '\\')
            {
                esc = true;
                j++;
                continue;
            }
            if(c == '"')
                break;
            j++;
        }
        if(j <= i || j >= header.size()) return "";
        return header.substr(i, j - i);
    };

    const std::string type = find_string("type");
    if(type == "error")
    {
        out.ok = false;
        out.error = find_string("error");
        return true;
    }

    out.ok = find_bool("ok", false);
    out.frame_id = find_int("frame_id", -1);
    out.timestamp = find_double("timestamp", -1.0);
    out.elapsed_ms = find_double("elapsed_ms", 0.0);
    out.profile_infer_ms = find_double("profile_infer_ms", 0.0);
    out.profile_post_ms = find_double("profile_post_ms", 0.0);
    out.error = find_string("error");

    if(!out.ok)
        return true;

    if(payload.empty())
    {
        out.ok = false;
        out.error = "empty_payload";
        return true;
    }

    cv::Mat png(1, static_cast<int>(payload.size()), CV_8U, payload.data());
    cv::Mat label_u16 = cv::imdecode(png, cv::IMREAD_UNCHANGED);
    if(label_u16.empty() || label_u16.type() != CV_16U)
    {
        out.ok = false;
        out.error = "decode_label_png_failed";
        return true;
    }

    cv::Mat label_s32;
    label_u16.convertTo(label_s32, CV_32S);
    label_s32 -= 1;
    label_s32.convertTo(out.label_map, CV_16S);
    return true;
}

} // namespace ORB_SLAM3
