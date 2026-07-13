#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "camera/Picamera2Camera.h"
#include "camera/CameraThread.h"          // ← เพิ่ม
#include "inference/YoloDetector.h"
#include "inference/SpeedSignClassifier.h"
#include "audio/NotificationManager.h"        // shared L4 (owned here, fed by both brains)
#include "audio/MomentaryAudioMap.h"          // Brain 2 — class -> wav
#include "decision/ShadowSpeedLimitPipeline.h"
#include "decision/BehaviorPolicyRouter.h"    // Brain 2 dispatch (after confirm)
#include "decision/MomentaryEngine.h"         // Brain 2 — momentary (non-speed) signs
#include "decision/NotificationArbiter.h"     // cross-brain attention scheduler (front door of L4)
#include "utils/Timer.h"
#include "utils/Types.h"
#include "vision/Draw.h"

static volatile std::sig_atomic_t g_running = 1;
void signal_handler(int) { g_running = 0; }

namespace fs = std::filesystem;
using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

// CooldownManager (legacy voter-input suppression) removed at cutover 2026-06-17.
// Anti-spam is now owned entirely by L3 (AnnouncementPolicy):
// CHANGE-always / REMINDER-on-fresh@reminder_cooldown / SuppressContinuation.

// ============================================================
// TemporalVoter — ไม่เปลี่ยนแปลง
// ============================================================
struct TemporalVoter {
    explicit TemporalVoter(int max_frames = 10, int early_confirm_votes = 4)
        : max_frames_(max_frames),
          early_confirm_votes_(early_confirm_votes) {}

    void update(const std::string& top_class) {
        history_.push_back(top_class);
        if (static_cast<int>(history_.size()) > max_frames_)
            history_.pop_front();
    }

    struct VoteResult {
        std::string winner;
        int winner_count = 0;
        int history_size = 0;
        bool confirmed   = false;
        std::map<std::string, int> votes;
    };

    VoteResult evaluate() const {
        VoteResult result;
        result.history_size = static_cast<int>(history_.size());
        for (const auto& cls : history_)
            if (!cls.empty()) result.votes[cls]++;
        if (result.votes.empty()) return result;

        std::string best_class;
        int best_count = 0;
        for (const auto& kv : result.votes) {
            if (kv.second > best_count) {
                best_count = kv.second;
                best_class = kv.first;
            } else if (kv.second == best_count) {
                for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
                    if (*it == kv.first)  { best_class = kv.first; break; }
                    if (*it == best_class) { break; }
                }
            }
        }

        result.winner       = best_class;
        result.winner_count = best_count;

        // Confirm ONLY on real evidence: the leader must reach early_confirm_votes_
        // within the rolling window. The old "window full (history >= max_frames)
        // → confirm the leader even with 1 vote" fallback was REMOVED 2026-07-13:
        // at 18 FPS a 10-frame window fills in ~0.56 s, so a single far-away
        // detection could force a confirm and cry-wolf on the first (K-less) Acquire.
        // Below threshold now = stay silent (winner=""). max_frames_ is still the
        // rolling-window size (a class has that many frames to reach the threshold).
        if (best_count >= early_confirm_votes_) {
            result.confirmed = true;
            return result;
        }
        result.winner    = "";
        result.confirmed = false;
        return result;
    }

    void reset() { history_.clear(); }
    const std::deque<std::string>& history() const { return history_; }

private:
    int max_frames_;
    int early_confirm_votes_;
    std::deque<std::string> history_;
};

// ============================================================

struct RollingAverage {
    explicit RollingAverage(int max_count) : max_count_(std::max(1, max_count)) {}

    void add(double value) {
        values_.push_back(value);
        sum_ += value;
        if (static_cast<int>(values_.size()) > max_count_) {
            sum_ -= values_.front();
            values_.pop_front();
        }
    }

    double value() const {
        if (values_.empty()) return 0.0;
        return sum_ / static_cast<double>(values_.size());
    }

private:
    int max_count_;
    std::deque<double> values_;
    double sum_ = 0.0;
};

struct TimingAverages {
    explicit TimingAverages(int window)
        : fps(window), output_fps(window), total(window), capture(window),
          capture_wait(window), queue_wait(window), latency(window), detect(window),
          preprocess(window), infer(window), postprocess(window), classify(window),
          result_wait(window), vote(window), render(window), draw(window) {}

    void add(const StageTimes& times, double fps_value) {
        fps.add(fps_value);
        total.add(times.total_ms);
        capture.add(times.capture_ms);
        capture_wait.add(times.capture_wait_ms);
        queue_wait.add(times.queue_wait_ms);
        latency.add(times.latency_ms);
        detect.add(times.detect_ms);
        preprocess.add(times.preprocess_ms);
        infer.add(times.infer_ms);
        postprocess.add(times.postprocess_ms);
        classify.add(times.classify_ms);
        result_wait.add(times.result_wait_ms);
        vote.add(times.vote_ms);
        render.add(times.render_ms);
        draw.add(times.draw_ms);
    }

    void add_output_fps(double fps_value) { output_fps.add(fps_value); }

    RollingAverage fps, output_fps, total, capture, capture_wait, queue_wait,
                   latency, detect, preprocess, infer, postprocess, classify,
                   result_wait, vote, render, draw;
};

struct PipelineCounters {
    uint64_t submitted_frames  = 0;
    uint64_t completed_outputs = 0;
    uint64_t dropped_frames    = 0;
    uint64_t pending_overwrites= 0;
    uint64_t result_overwrites = 0;
};

static void save_debug_frame(
    const cv::Mat& frame_bgr,
    const std::vector<Detection>& detections,
    const AppConfig& cfg,
    int frame_index
) {
    const std::string bucket = detections.empty() ? "miss" : "hit";
    const fs::path dir = fs::path(cfg.save_dir) / bucket;
    fs::create_directories(dir);

    std::string label = "none";
    float conf = 0.0f;
    if (!detections.empty()) {
        const auto best = std::max_element(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) {
                return a.confidence < b.confidence;
            });
        label = best->class_name;
        conf  = best->confidence;
    }

    const std::string filename =
        cv::format("%06d_%s_%.2f.jpg", frame_index, label.c_str(), conf);
    cv::imwrite((dir / filename).string(), frame_bgr);
}

static AppConfig parse_args(int argc, char** argv) {
    AppConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };

        if      (key == "--model" || key == "--ncnn-param") cfg.model_path       = value(key);
        else if (key == "--ncnn-bin")        cfg.ncnn_bin_path    = value(key);
        else if (key == "--input-name")      cfg.ncnn_input_name  = value(key);
        else if (key == "--output-name")     cfg.ncnn_output_name = value(key);
        else if (key == "--imgsz")           cfg.imgsz            = std::stoi(value(key));
        else if (key == "--conf")            cfg.conf_threshold   = std::stof(value(key));
        else if (key == "--iou")             cfg.iou_threshold    = std::stof(value(key));
        else if (key == "--threads")         cfg.threads          = std::stoi(value(key));
        else if (key == "--camera-width")    cfg.camera_width     = std::stoi(value(key));
        else if (key == "--camera-height")   cfg.camera_height    = std::stoi(value(key));
        else if (key == "--camera-fps")      cfg.camera_fps       = std::stoi(value(key));
        else if (key == "--avg-window")      cfg.avg_window       = std::stoi(value(key));
        else if (key == "--top-k")           cfg.terminal_top_k   = std::stoi(value(key));
        else if (key == "--save-dir")        cfg.save_dir         = value(key);
        else if (key == "--save-every")      cfg.save_every       = std::stoi(value(key));
        else if (key == "--hide-window")     cfg.hide_window      = true;
        else if (key == "--show-window")     cfg.hide_window      = false;
        else if (key == "--no-draw")       { cfg.no_draw = true; cfg.hide_window = true; }
        else if (key == "--save-frames")     cfg.save_frames      = true;
        else if (key == "--vulkan")          cfg.use_vulkan       = true;
        else if (key == "--no-packing")      cfg.use_packing      = false;
        else if (key == "--shadow")          cfg.shadow           = true;
        else if (key == "--shadow-verbose")  cfg.shadow_verbose   = true;
        else if (key == "--shadow-k")        cfg.shadow_k             = std::stoi(value(key));
        else if (key == "--shadow-rearm-ms") cfg.shadow_rearm_ms      = std::stoi(value(key));
        else if (key == "--shadow-reminder-sec") cfg.shadow_reminder_sec = std::stoi(value(key));
        else if (key == "--early-confirm")   cfg.early_confirm_votes = std::stoi(value(key));
        else if (key == "--audio")           cfg.audio            = true;
        else if (key == "--audio-dir")       cfg.audio_dir        = value(key);
        else if (key == "--audio-device")    cfg.audio_device     = value(key);
        else if (key == "--async-detect")    cfg.async_detect     = true;
        // ← flag ใหม่: --async-camera เปิด CameraThread
        else if (key == "--async-camera")    cfg.async_camera     = true;
        else if (key == "--cls-param")       cfg.cls_param_path   = value(key);
        else if (key == "--cls-bin")         cfg.cls_bin_path     = value(key);
        else if (key == "--cls-min-conf")    cfg.cls_min_conf     = std::stof(value(key));
        else if (key == "--save-roi-debug")  cfg.roi_debug_dir    = value(key);
        else if (key == "--help" || key == "-h") {
            std::cout << "Usage: ./app --ncnn-param <param> --ncnn-bin <bin> [options]\n"
                      << "  --async-detect   enable async detector thread\n"
                      << "  --async-camera   enable dedicated camera thread (double buffer)\n"
                      << "  --show-window    enable OpenCV GUI window (default: headless)\n"
                      << "  --no-draw        skip overlay drawing for lowest CPU use\n"
                      << "  --shadow         (no-op; L1/L2/L3 pipeline is the authority since cutover)\n"
                      << "  --shadow-verbose log SUPPRESS events from the L1/L2/L3 pipeline\n"
                      << "  --shadow-k <n>            L2 K-hysteresis (default 2)\n"
                      << "  --shadow-rearm-ms <ms>    L1 re-arm timeout (default 600)\n"
                      << "  --shadow-reminder-sec <s> L3 reminder cooldown (default 180)\n"
                      << "  --early-confirm <n>       voter votes needed to confirm (default 4)\n"
                      << "  --audio          play audio on speed announce\n"
                      << "  --audio-dir <d>     wav directory (default ../assets/audio)\n"
                      << "  --audio-device <d>  ALSA device for aplay (default plughw:0,0)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (cfg.avg_window < 1)     throw std::runtime_error("--avg-window must be >= 1");
    if (cfg.terminal_top_k < 1) throw std::runtime_error("--top-k must be >= 1");
    if (cfg.save_every < 1)     throw std::runtime_error("--save-every must be >= 1");
    return cfg;
}

struct DetectionResult {
    int frame_index = 0;
    cv::Mat frame_bgr;
    StageTimes times;
    TimePoint start_time;
    TimePoint submit_time;
    TimePoint inference_start_time;
    TimePoint result_ready_time;
    std::vector<Detection> detections;
    bool valid = false;
};

struct DecisionResult {
    std::string top_class;
    float top_conf = 0.0f;
    TemporalVoter::VoteResult vote;
};

// ============================================================
// capture_frame — sync version (ใช้เมื่อ async_camera=false)
// ไม่เปลี่ยนแปลงจากเดิม
// ============================================================
static bool capture_frame(
    Picamera2Camera& camera,
    cv::Mat& frame_bgr,
    StageTimes& times
) {
    Timer timer;
    timer.tic();
    if (!camera.read(frame_bgr)) {
        times.capture_ms = timer.toc_ms();
        times.capture_wait_ms = times.capture_ms;
        std::cerr << "Failed to read frame\n";
        return false;
    }
    times.capture_ms = timer.toc_ms();
    times.capture_wait_ms = times.capture_ms;
    return true;
}

// ============================================================
// capture_frame_async — async version (ใช้เมื่อ async_camera=true)
//
// แทนที่จะ block รอ camera.read() ใน main thread
// เรียก get_latest_frame() จาก CameraThread แทน
//
// ความแตกต่างด้านเวลา:
//   sync:  block ~30ms ทุก call
//   async: ~0.1ms สำหรับ clone (ถ้ามี frame พร้อม)
//          หรือ busy-wait เล็กน้อยถ้ายังไม่มี frame (ช่วง startup)
//
// capture_wait_ms ใน async mode จะวัด:
//   เวลาที่ main thread ใช้รอจนได้ frame จาก CameraThread
//   ถ้า CameraThread เร็วกว่า main → ~0ms
//   ถ้า main เร็วกว่า CameraThread (ไม่น่าเกิด) → รอ
// ============================================================
static bool capture_frame_async(
    CameraThread& cam_thread,
    CameraFrame& cam_frame,
    StageTimes& times
) {
    Timer timer;
    timer.tic();

    // busy-wait เล็กน้อยถ้ายังไม่มี frame (เกิดแค่ตอน startup)
    // ในทางปกติ CameraThread produce ก่อน main consume
    // ดังนั้น get_latest_frame() จะสำเร็จทันทีใน iteration แรก
    while (!cam_thread.get_latest_frame(cam_frame)) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        if (!g_running) return false;
    }

    times.capture_ms      = cam_thread.capture_ms(); // เวลาที่กล้องใช้ capture จริง
    times.capture_wait_ms = timer.toc_ms();           // เวลาที่ main รอได้ frame
    return true;
}

static DetectionResult run_detection(
    YoloDetector& detector,
    const cv::Mat& frame_bgr,
    StageTimes& times
) {
    Timer timer;
    timer.tic();

    DetectionResult result;
    result.detections = detector.detect(
        frame_bgr,
        &times.preprocess_ms,
        &times.infer_ms,
        &times.postprocess_ms,
        &times.classify_ms
    );

    times.detect_ms = timer.toc_ms();
    result.times = times;
    result.valid = true;
    return result;
}

static DecisionResult run_decision(
    const std::vector<Detection>& detections,
    const cv::Mat& frame_bgr,
    TemporalVoter& voter,
    SpeedSignClassifier* classifier,
    std::map<std::string, BestROI>& roi_by_class,
    ShadowSpeedLimitPipeline& pipeline,   // L1/L2/L3 — AUTHORITY ([SHADOW][L3])
    MomentaryEngine& momentary,           // Brain 2 — non-speed momentary signs
    NotificationManager& notifier,        // shared L4 (effect: ส่งเสียงเมื่อ Arbiter อนุญาต)
    NotificationArbiter& arbiter,         // cross-brain scheduler: ตัดสิน play/preempt/drop ด้วย rank
    bool shadow_enabled,
    const std::string& roi_debug_dir,   // "" = disabled, path = save crops
    int frame_index,
    StageTimes& times
) {
    Timer timer;
    timer.tic();

    DecisionResult result;
    if (!detections.empty()) {
        const auto best = std::max_element(
            detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) {
                return a.confidence < b.confidence;
            });
        result.top_class = best->class_name;
        result.top_conf  = best->confidence;

        // BestROI tracking ต่อ class: เก็บ frame+box ที่ YOLO conf สูงสุดใน window
        // ของแต่ละ class แยกกัน → classifier ได้ ROI ของ winning class เสมอ ไม่ปนข้าม class
        // (cutover 2026-06-17: ตัด cooldown gate ออก — track ทุกเฟรมที่เป็น speed sign)
        if (SpeedSignClassifier::speed_sign_group().count(result.top_class)) {
            roi_by_class[result.top_class].update(frame_bgr, best->box, result.top_conf, frame_index);
        }
    }

    // cutover 2026-06-17: voter ป้อนด้วย top_class ดิบเสมอ (ไม่ผ่าน voter-input cooldown อีก)
    // → confirm ไหลตาม detection cadence จริง; anti-spam ย้ายไป L3 ทั้งหมด
    voter.update(result.top_class);
    result.vote = voter.evaluate();

    if (!result.top_class.empty()) {
        std::cout << "[DET F" << frame_index << "]"
                  << " class=" << result.top_class
                  << " conf=" << cv::format("%.2f", result.top_conf)
                  << " | votes={";
        for (const auto& kv : result.vote.votes)
            std::cout << kv.first << ":" << kv.second << " ";
        std::cout << "} buf=" << result.vote.history_size << "/10"
                  << " winner=" << (result.vote.winner.empty() ? "none" : result.vote.winner)
                  << " winner_count=" << result.vote.winner_count
                  << "\n" << std::flush;
    }

    // CLS-corrected value ของ confirm นี้ (อ่านโดย shadow pipeline ด้านล่าง)
    // ว่าง = เฟรมนี้ไม่ confirmed; set เฉพาะใน block ข้างล่างหลัง CLS correction
    std::string confirmed_value;

    if (result.vote.confirmed) {
        std::string output = result.vote.winner;
        voter.reset();

        // Confirm-then-Classify:
        // classifier รันครั้งเดียวตอน confirm เท่านั้น
        // ใช้ best_roi (frame ที่ YOLO conf สูงสุดจาก window)
        // ไม่ใช่ frame ปัจจุบัน ซึ่งอาจจะไม่ใช่ frame ที่ดีที่สุด
        auto roi_it = roi_by_class.find(output);   // output = voter winner
        if (classifier &&
            SpeedSignClassifier::speed_sign_group().count(output) &&
            roi_it != roi_by_class.end() && roi_it->second.valid)
        {
            const BestROI& win_roi = roi_it->second;   // ROI ของ winning class เท่านั้น
            auto t_cls0 = std::chrono::high_resolution_clock::now();
            float cls_conf = 0.0f;
            const std::string cls_result = classifier->classify(
                win_roi.frame_bgr, win_roi.box, &cls_conf,
                roi_debug_dir,          // ส่ง path → save crop ถ้าไม่ว่าง
                output,                 // yolo class (voter winner)
                win_roi.frame_idx,      // frame ที่ ROI มาจาก
                win_roi.yolo_conf       // conf ของ frame นั้น
            );
            auto t_cls1 = std::chrono::high_resolution_clock::now();
            times.classify_ms = std::chrono::duration<double, std::milli>(
                t_cls1 - t_cls0
            ).count();

            if (!cls_result.empty()) {
                if (cls_result != output) {
                    std::cout << "[CLS] voter=" << output
                              << " -> classifier=" << cls_result
                              << " conf=" << cv::format("%.2f", cls_conf)
                              << " best_roi_F" << win_roi.frame_idx
                              << " yolo_conf=" << cv::format("%.2f", win_roi.yolo_conf)
                              << "\n" << std::flush;
                }
                output = cls_result;
            } else {
                std::cout << "[CLS] low_conf=" << cv::format("%.2f", cls_conf)
                          << " keep_voter=" << output << "\n" << std::flush;
            }

            // [CONFIRM] evidence snapshot (2026-07-13): vote strength + spread +
            // ROI size (distance proxy) + CLS conf. Shows how strong each confirm
            // was and how long a real sign took to reach the vote threshold.
            std::cout << "[CONFIRM F" << frame_index << "]"
                      << " voter=" << result.vote.winner
                      << " count=" << result.vote.winner_count
                      << " votes={";
            for (const auto& kv : result.vote.votes)
                std::cout << kv.first << ":" << kv.second << " ";
            std::cout << "} cls=" << output
                      << " cls_conf=" << cv::format("%.2f", cls_conf)
                      << " roi=" << win_roi.box.width << "x" << win_roi.box.height
                      << " yolo=" << cv::format("%.2f", win_roi.yolo_conf)
                      << "\n" << std::flush;
        }

        roi_by_class.clear();   // เคลียร์ ROI ทุก class — เริ่ม window ใหม่ พร้อม voter.reset()

        // cutover 2026-06-17: legacy [CONFIRMED] + cooldown.activate() removed.
        // CLS correction above stays (computes the value); the announce/suppress
        // decision is now entirely L3's (see pipeline.tick below).
        confirmed_value = output;   // CLS-corrected → ป้อน L1/L2/L3 pipeline (speed only)

        // ── Brain 2 dispatch (BehaviorPolicyRouter) ──────────────────────
        // non-speed confirmed sign → MomentaryEngine. LOG-ONLY for now: audio +
        // cross-brain arbitration come with refactor #2 + Notification Arbiter.
        // (speed handled by pipeline.tick below; None = unknown class → ignore)
        if (BehaviorPolicyRouter::route(output) == BehaviorPolicyRouter::Brain::Momentary) {
            const TimePoint now = Clock::now();
            const auto m = momentary.onConfirmed(output, now);
            if (MomentaryEngine::is_announce(m.decision)) {
                // engine ตัดสิน "ควรพูดไหม" แล้ว → Arbiter ตัดสิน "ได้ช่องไหม" ด้วย rank ร่วมสองสมอง
                const auto ar = arbiter.submit(m.attention_rank,
                                               MomentaryAudioMap::filename(m.cls), now);
                const char* act =
                    ar.decision == NotificationArbiter::Decision::Play    ? "PLAY"    :
                    ar.decision == NotificationArbiter::Decision::Preempt ? "PREEMPT" : "DROP";
                std::cout << "[MOMENTARY] " << act << " " << m.cls
                          << " rank=" << m.attention_rank
                          << " F" << frame_index << "\n" << std::flush;
                // Play = ต่อคิว, Preempt (safety แทรก) = ตัดคลิป speed ที่เล่นอยู่กลางคัน
                if      (ar.decision == NotificationArbiter::Decision::Play)    notifier.submit(ar.filename);
                else if (ar.decision == NotificationArbiter::Decision::Preempt) notifier.preempt(ar.filename);
            }
            // SUPPRESS = เงียบ (ไม่ log กัน spam ตอนป้ายค้างในเฟรม)
        }
    }

    times.vote_ms = timer.toc_ms();

    // ── L1/L2/L3 pipeline ([SHADOW][L3]) — AUTHORITY after cutover ──
    // presence = RAW class-agnostic speed-sign presence จาก detections ทั้งหมด
    //   (ไม่ผ่าน cooldown, ไม่ใช่แค่ top_class — เป็นข้อเท็จจริงเชิง perception)
    if (shadow_enabled) {
        bool speed_presence = false;
        for (const auto& det : detections) {
            if (SpeedSignClassifier::speed_sign_group().count(det.class_name)) {
                speed_presence = true;
                break;
            }
        }
        pipeline.tick(speed_presence, result.vote.confirmed,
                      confirmed_value, frame_index, Clock::now());
    }

    // Re-delivery (cross-brain): ช่องลำโพงว่างจริง (L4 is_idle) + มี speed CHANGE ที่ถูก safety
    //   ตัดค้างอยู่ → replay จาก belief ปัจจุบันของ L2. เช็คทุกเฟรม (main thread เท่านั้น).
    if (notifier.is_idle() && arbiter.poll(Clock::now())) {
        pipeline.redeliver(Clock::now());
    }

    return result;
}

static bool render_output(
    const cv::Mat& frame_bgr,
    const std::vector<Detection>& detections,
    cv::Mat& canvas_bgr,
    const AppConfig& cfg,
    const TimingAverages& avg,
    StageTimes& times,
    int frame_index
) {
    Timer timer;
    timer.tic();

    times.draw_ms = 0.0;
    if (!cfg.no_draw) {
        Timer draw_timer;
        draw_timer.tic();
        canvas_bgr = frame_bgr.clone();
        draw_phase1_overlay(
            canvas_bgr, detections, times,
            avg.fps.value(), cfg.imgsz, cfg.conf_threshold
        );
        times.draw_ms = draw_timer.toc_ms();
    }

    if (cfg.save_frames && frame_index % cfg.save_every == 0) {
        save_debug_frame(frame_bgr, detections, cfg, frame_index);
    }

    if (!cfg.hide_window && !cfg.no_draw) {
        cv::imshow("Detection", canvas_bgr);
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') {
            times.render_ms = timer.toc_ms();
            return false;
        }
    }

    times.render_ms = timer.toc_ms();
    return true;
}

static float top_detection_confidence(const std::vector<Detection>& detections) {
    float top_conf = 0.0f;
    for (const auto& det : detections) {
        top_conf = std::max(top_conf, det.confidence);
    }
    return top_conf;
}

static void print_performance(
    const TimingAverages& avg,
    size_t detection_count,
    float top_conf,
    const PipelineCounters& counters
) {
    std::cout << "\rFPS: " << cv::format("%.1f", avg.fps.value())
              << " | out_fps: " << cv::format("%.1f", avg.output_fps.value())
              << " | total: "   << cv::format("%.1f", avg.total.value())       << "ms"
              << " | latency: " << cv::format("%.1f", avg.latency.value())     << "ms"
              << " | cap: "     << cv::format("%.1f", avg.capture.value())     << "ms"
              << " | cap_wait: "<< cv::format("%.1f", avg.capture_wait.value())<< "ms"
              << " | q_wait: "  << cv::format("%.1f", avg.queue_wait.value())  << "ms"
              << " | pre: "     << cv::format("%.1f", avg.preprocess.value())  << "ms"
              << " | infer: "   << cv::format("%.1f", avg.infer.value())       << "ms"
              << " | post: "    << cv::format("%.1f", avg.postprocess.value()) << "ms"
              << " | cls: "     << cv::format("%.1f", avg.classify.value())    << "ms"
              << " | res_wait: "<< cv::format("%.1f", avg.result_wait.value()) << "ms"
              << " | vote: "    << cv::format("%.1f", avg.vote.value())        << "ms"
              << " | render: "  << cv::format("%.1f", avg.render.value())      << "ms"
              << " | det: "     << detection_count
              << " | conf: "    << cv::format("%.2f", top_conf)
              << " | drop: "    << counters.dropped_frames
              << " | ow: "      << counters.pending_overwrites
              << "          "
              << std::flush;
}

// ============================================================
// AsyncDetectionWorker — ไม่เปลี่ยนแปลงจากเดิม
// ============================================================
class AsyncDetectionWorker {
public:
    explicit AsyncDetectionWorker(YoloDetector& detector)
        : detector_(detector),
          worker_(&AsyncDetectionWorker::run, this) {}

    ~AsyncDetectionWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    void submit_latest(
        int frame_index,
        const cv::Mat& frame_bgr,
        const StageTimes& times,
        TimePoint start_time
    ) {
        DetectionResult next;
        next.frame_index  = frame_index;
        next.frame_bgr    = frame_bgr.clone();
        next.times        = times;
        next.start_time   = start_time;
        next.submit_time  = Clock::now();
        next.valid        = true;

        std::lock_guard<std::mutex> lock(mutex_);
        if (has_pending_) {
            ++counters_.pending_overwrites;
            ++counters_.dropped_frames;
        }
        pending_     = std::move(next);
        has_pending_ = true;
        ++counters_.submitted_frames;
        cv_.notify_one();
    }

    bool try_take_result(DetectionResult& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_error_) std::rethrow_exception(worker_error_);
        if (!has_result_)  return false;
        result    = std::move(result_);
        result_   = DetectionResult{};
        has_result_ = false;
        return true;
    }

    PipelineCounters counters() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return counters_;
    }

private:
    void run() {
        while (true) {
            DetectionResult job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || has_pending_; });
                if (stop_ && !has_pending_) return;
                job = std::move(pending_);
                pending_     = DetectionResult{};
                has_pending_ = false;
            }

            try {
                job.inference_start_time = Clock::now();
                job.times.queue_wait_ms  = std::chrono::duration<double, std::milli>(
                    job.inference_start_time - job.submit_time
                ).count();

                DetectionResult detected = run_detection(detector_, job.frame_bgr, job.times);
                job.times      = detected.times;
                job.detections = std::move(detected.detections);
                job.result_ready_time = Clock::now();
                job.valid = true;

                std::lock_guard<std::mutex> lock(mutex_);
                if (has_result_) {
                    ++counters_.result_overwrites;
                    ++counters_.dropped_frames;
                }
                result_     = std::move(job);
                has_result_ = true;
                ++counters_.completed_outputs;
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                worker_error_ = std::current_exception();
                stop_ = true;
                return;
            }
        }
    }

    YoloDetector& detector_;
    std::thread worker_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    DetectionResult pending_;
    DetectionResult result_;
    bool has_pending_ = false;
    bool has_result_  = false;
    bool stop_        = false;
    PipelineCounters counters_;
    std::exception_ptr worker_error_;
};

int main(int argc, char** argv) {
    try {
        std::signal(SIGINT, signal_handler);
        AppConfig cfg = parse_args(argc, argv);

        // สร้าง pipeline mode string สำหรับ log
        std::string pipeline_mode = "Sync";
        if (cfg.async_camera && cfg.async_detect)
            pipeline_mode = "AsyncCamera + AsyncDetect";
        else if (cfg.async_camera)
            pipeline_mode = "AsyncCamera + SyncDetect";
        else if (cfg.async_detect)
            pipeline_mode = "SyncCamera + AsyncDetect";

        std::cout << "===== Traffic Sign Detection =====\n"
                  << "Pipeline: " << pipeline_mode << "\n"
                  << "Voting  : window=10, early_confirm=4\n";

        if (!cfg.roi_debug_dir.empty()) {
            std::cout << "ROI debug: ON → " << cfg.roi_debug_dir << "\n";
        }
        std::cout << "\n";

        YoloDetector detector(
            cfg.model_path, cfg.ncnn_bin_path,
            cfg.imgsz, cfg.conf_threshold, cfg.iou_threshold,
            cfg.threads, cfg.ncnn_input_name, cfg.ncnn_output_name,
            cfg.use_vulkan, cfg.use_packing
        );

        Picamera2Camera camera(cfg.camera_width, cfg.camera_height, cfg.camera_fps);

        if (!cfg.hide_window) {
            cv::namedWindow("Detection", cv::WINDOW_NORMAL);
        }

        TimingAverages avg(cfg.avg_window);
        TemporalVoter   voter(10, cfg.early_confirm_votes);

        // ROI ต่อ class — เก็บ crop คมชัดที่สุดของแต่ละ class ระหว่าง voting window
        // key = class name → classifier อ่าน roi_by_class[winner] ตอน confirm
        // clear ทั้งหมดหลัง confirm (พร้อม voter.reset())
        std::map<std::string, BestROI> roi_by_class;

        // raw pointer ไปยัง classifier (ถ้ามี) สำหรับส่งเข้า run_decision
        // ownership ยังอยู่ที่ detector ผ่าน set_classifier()
        // ดังนั้นใช้ raw pointer ที่นี่ได้อย่างปลอดภัย
        SpeedSignClassifier* classifier_ptr = nullptr;

        // ── Async Camera Thread (ถ้าเปิด) ──────────────────────
        // CameraThread เริ่มทำงานทันทีที่สร้าง
        // camera.read() ย้ายไปทำใน thread แยก
        // main thread จะเรียก get_latest_frame() แทน
        std::unique_ptr<CameraThread> cam_thread;
        if (cfg.async_camera) {
            cam_thread = std::make_unique<CameraThread>(camera);
            std::cout << "[CameraThread] started (double buffer)\n";
        }

        // Attach classifier ถ้ามี --cls-param
        if (!cfg.cls_param_path.empty() && !cfg.cls_bin_path.empty()) {
            auto cls = std::make_unique<SpeedSignClassifier>(
                cfg.cls_param_path, cfg.cls_bin_path,
                cfg.threads, cfg.use_vulkan, cfg.use_packing,
                cfg.cls_min_conf
            );
            // get() raw pointer ก่อน move — ปลอดภัยเพราะ
            // detector เป็น owner ผ่าน set_classifier()
            // classifier_ptr ใช้แค่อ่าน ไม่มี ownership
            classifier_ptr = cls.get();
            detector.set_classifier(std::move(cls));
            std::cout << "[Classifier] SpeedSignClassifier loaded (Confirm-then-Classify mode)\n"
                      << "  param: " << cfg.cls_param_path << "\n"
                      << "  bin:   " << cfg.cls_bin_path << "\n"
                      << "  min_conf: " << cfg.cls_min_conf << "\n";
        }

        // Shared L4 — ONE audio output (one thread, one latest-wins slot). Owned here so that
        // both Brain 1 (speed, below) and the future Brain 2 / Notification Arbiter feed the
        // SAME channel. enabled=false (no --audio) → NotificationManager is a no-op, no thread.
        // Must outlive `pipeline` (declared first; same scope).
        NotificationManager notifier(cfg.audio_dir, cfg.audio_device, cfg.audio);

        // Notification Arbiter — cross-brain attention scheduler นั่งหน้า shared L4.
        // ทั้ง speed (pipeline) และ momentary (run_decision) route ผ่านตัวนี้ → ตัดสิน
        // play/preempt/drop ด้วย attention_rank ร่วมกัน (แทน interim direct-submit / last-wins).
        // busy-window = nominal_clip default (2.5s, scaffold); feedback จริง + re-deliver มากับ #3.
        // Must outlive `pipeline` (declared first; same scope).
        NotificationArbiter arbiter;

        // L1/L2/L3 pipeline — [SHADOW][L3], the speed AUTHORITY after cutover 2026-06-17.
        // Feeds the shared L4 via the Arbiter (no longer owns L4, no longer submits direct).
        // tick ถูกเรียกใน run_decision (main thread เท่านั้น) → L1/L2/L3 ไม่ต้อง lock
        ShadowSpeedLimitPipeline pipeline(
            cfg.shadow_k,
            std::chrono::milliseconds(cfg.shadow_rearm_ms),
            std::chrono::seconds(cfg.shadow_reminder_sec),
            cfg.shadow_verbose,
            &notifier,
            &arbiter
        );

        // Brain 2 — Momentary engine (non-speed signs). LOG-ONLY for now: routed by
        // BehaviorPolicyRouter in run_decision; audio/arbitration land with refactor #2.
        MomentaryEngine momentary;

        if (cfg.shadow) {
            std::cout << "[SPEED] L1/L2/L3 pipeline ON (authority)"
                      << " (K=" << cfg.shadow_k
                      << ", rearm=" << cfg.shadow_rearm_ms << "ms"
                      << ", reminder=" << cfg.shadow_reminder_sec << "s)\n";
        }
        if (cfg.audio) {
            std::cout << "[AUDIO] ON (dir=" << cfg.audio_dir
                      << ", device=" << cfg.audio_device << ")\n";
            if (!cfg.shadow) {
                std::cout << "[AUDIO] WARNING: speed pipeline is off → silent\n";
            }
        }

        std::unique_ptr<AsyncDetectionWorker> async_detector;
        if (cfg.async_detect) {
            async_detector = std::make_unique<AsyncDetectionWorker>(detector);
        }

        cv::Mat frame_bgr, canvas_bgr;
        CameraFrame cam_frame;           // ใช้กับ async camera
        StageTimes times;
        int frame_index = 0;
        TimePoint last_output_time;
        bool has_last_output_time = false;
        PipelineCounters sync_counters;

        auto update_output_metrics = [&](
            StageTimes& output_times,
            const TimePoint& frame_start,
            const TimePoint& result_ready_time,
            const PipelineCounters& counters,
            size_t detection_count,
            float top_conf
        ) {
            const TimePoint output_time = Clock::now();
            if (result_ready_time != TimePoint{}) {
                output_times.result_wait_ms = std::chrono::duration<double, std::milli>(
                    output_time - result_ready_time
                ).count();
            }
            output_times.latency_ms = std::chrono::duration<double, std::milli>(
                output_time - frame_start
            ).count();
            output_times.total_ms = output_times.latency_ms;

            double output_fps = 0.0;
            if (has_last_output_time) {
                const double interval_ms = std::chrono::duration<double, std::milli>(
                    output_time - last_output_time
                ).count();
                if (interval_ms > 0.0) output_fps = 1000.0 / interval_ms;
            }
            last_output_time      = output_time;
            has_last_output_time  = true;

            avg.add(output_times, output_fps);
            avg.add_output_fps(output_fps);
            print_performance(avg, detection_count, top_conf, counters);
        };

        // ================================================================
        // MAIN LOOP
        //
        // 4 modes รวมกัน:
        //   async_camera=false, async_detect=false → sync ทั้งหมด (เดิม)
        //   async_camera=false, async_detect=true  → async detect (เดิม)
        //   async_camera=true,  async_detect=false → camera async, detect sync
        //   async_camera=true,  async_detect=true  → ทั้งคู่ async ← target mode
        //
        // ================================================================

        while (g_running) {

            // ── Step 1: Get Frame ───────────────────────────────────────
            // ถ้า async_camera: main thread ไม่ block รอ camera.read()
            //   → ดึง frame ล่าสุดจาก CameraThread แทน (~0ms)
            //   → capture_wait_ms จะเป็น ~0ms แทน ~30ms
            // ถ้า sync: block เหมือนเดิม

            ++frame_index;
            const TimePoint frame_start = Clock::now();
            times = StageTimes{};

            bool got_frame = false;
            if (cfg.async_camera) {
                got_frame = capture_frame_async(*cam_thread, cam_frame, times);
                if (got_frame) {
                    frame_bgr = cam_frame.frame_bgr;
                    // ปรับ frame_start ให้ตรงกับเวลาที่กล้อง capture จริง
                    // เพื่อให้ latency_ms วัดตั้งแต่ captured_at ไม่ใช่ตั้งแต่ main thread เริ่ม loop
                    // → latency จะ accurate กว่า
                }
            } else {
                got_frame = capture_frame(camera, frame_bgr, times);
            }

            if (!got_frame) continue;

            // ── Step 2: Submit to Detector ──────────────────────────────
            if (cfg.async_detect) {
                // ตรวจว่ามี result พร้อมก่อน submit frame ใหม่
                // ทำให้ main thread ไม่ idle รอ result
                DetectionResult detection;
                if (async_detector->try_take_result(detection)) {
                    times = detection.times;
                    run_decision(detection.detections, detection.frame_bgr,
                                 voter, classifier_ptr, roi_by_class,
                                 pipeline, momentary, notifier, arbiter, cfg.shadow,
                                 cfg.roi_debug_dir,
                                 detection.frame_index, times);

                    const bool keep_running = render_output(
                        detection.frame_bgr, detection.detections,
                        canvas_bgr, cfg, avg, times, detection.frame_index
                    );

                    update_output_metrics(
                        times, detection.start_time, detection.result_ready_time,
                        async_detector->counters(), detection.detections.size(),
                        top_detection_confidence(detection.detections)
                    );

                    if (!keep_running) break;
                }

                // submit frame ใหม่เข้า detector
                // ถ้ามี pending อยู่แล้วจะ overwrite (latest-frame policy)
                const TimePoint submit_start = cfg.async_camera
                    ? cam_frame.captured_at : frame_start;
                async_detector->submit_latest(frame_index, frame_bgr, times, submit_start);

            } else {
                // Sync detection
                const TimePoint detect_start = cfg.async_camera
                    ? cam_frame.captured_at : frame_start;

                DetectionResult detection = run_detection(detector, frame_bgr, times);
                detection.frame_index      = frame_index;
                detection.frame_bgr        = frame_bgr;
                detection.start_time       = detect_start;
                detection.result_ready_time = Clock::now();

                run_decision(detection.detections, frame_bgr,
                             voter, classifier_ptr, roi_by_class,
                             pipeline, momentary, notifier, arbiter, cfg.shadow,
                             cfg.roi_debug_dir,
                             frame_index, times);

                const bool keep_running = render_output(
                    frame_bgr, detection.detections, canvas_bgr,
                    cfg, avg, times, frame_index
                );

                ++sync_counters.submitted_frames;
                ++sync_counters.completed_outputs;
                update_output_metrics(
                    times, detection.start_time, detection.result_ready_time,
                    sync_counters, detection.detections.size(),
                    top_detection_confidence(detection.detections)
                );

                if (!keep_running) break;
            }
        }

        std::cout << "\nStopped.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
}
