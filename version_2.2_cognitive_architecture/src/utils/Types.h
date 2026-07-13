#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

struct Detection {
    cv::Rect box;
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0f;
};


// ============================================================
// BestROI — เก็บ crop ที่ YOLO confidence สูงสุดระหว่าง voting
//
// ใช้ส่งให้ classifier ตอน voting confirm แทนที่จะ classify ทุก frame
// ทำให้ classifier ได้รับ crop ที่คมชัดและ YOLO มั่นใจที่สุด
// ============================================================
struct BestROI {
    cv::Mat  frame_bgr;    // frame เต็ม (ไม่ crop) เพื่อ classifier จะ crop เอง
    cv::Rect box;          // box ใน frame_bgr coordinates
    float    yolo_conf = 0.0f;
    int      frame_idx = -1;
    bool     valid     = false;

    void update(const cv::Mat& frame, const cv::Rect& b, float conf, int fidx) {
        if (conf > yolo_conf) {
            frame_bgr = frame.clone();
            box       = b;
            yolo_conf = conf;
            frame_idx = fidx;
            valid     = true;
        }
    }

    void reset() {
        frame_bgr = cv::Mat{};
        box       = cv::Rect{};
        yolo_conf = 0.0f;
        frame_idx = -1;
        valid     = false;
    }
};

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int input_size = 0;
};

struct StageTimes {
    double capture_ms = 0.0;
    double capture_wait_ms = 0.0;
    double queue_wait_ms = 0.0;
    double detect_ms = 0.0;
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double result_wait_ms = 0.0;
    double classify_ms = 0.0;
    double vote_ms = 0.0;
    double render_ms = 0.0;
    double draw_ms = 0.0;
    double latency_ms = 0.0;
    double total_ms = 0.0;
};

struct AppConfig {
    std::string model_path = "../src/models/detection/yolo11n/model.ncnn.param";
    std::string ncnn_bin_path = "../src/models/detection/yolo11n/model.ncnn.bin";
    std::string ncnn_input_name = "in0";
    std::string ncnn_output_name = "out0";
    int imgsz = 512;
    float conf_threshold = 0.45f;
    float iou_threshold = 0.45f;
    int threads = 2;
    int camera_width = 960;
    int camera_height = 560;
    int camera_fps = 30;
    bool hide_window = true;
    bool no_draw = false;
    bool use_vulkan = false;
    bool use_packing = true;
    bool async_detect = false;
    bool async_camera = false;
    int  early_confirm_votes = 4;       // --early-confirm: โหวตขั้นต่ำที่ voter ต้องได้ถึง confirm (window=10, ไม่มี fallback)

    // L1/L2/L3 pipeline (Step 3) — AUTHORITY after cutover 2026-06-17 (default ON)
    bool shadow              = true;    // legacy --shadow flag kept as no-op; pipeline is now the authority
    bool shadow_verbose      = false;   // --shadow-verbose: log SUPPRESS ของ shadow ด้วย
    int  shadow_k            = 2;       // --shadow-k: K ของ L2 (>=1)
    int  shadow_rearm_ms     = 600;     // --shadow-rearm-ms: rearm_after ของ L1
    int  shadow_reminder_sec = 180;     // --shadow-reminder-sec: reminder cooldown ของ L3

    // L4 NotificationManager (audio) — log-only เหมือนกัน, ตาม shadow decisions
    bool        audio        = false;            // --audio: เปิดเสียง (ต้องใช้คู่ --shadow)
    std::string audio_dir    = "../assets/audio";// --audio-dir: โฟลเดอร์ .wav
    std::string audio_device = "plughw:0,0";     // --audio-device: ALSA device ของ aplay

    // Speed sign classifier (optional)
    std::string cls_param_path = "../src/models/classification/speed_classifier/model.ncnn.param";
    std::string cls_bin_path   = "../src/models/classification/speed_classifier/model.ncnn.bin";
    float cls_min_conf         = 0.70f;
    std::string roi_debug_dir  = "";   // ถ้าไม่ว่าง → save ROI crops
    int avg_window = 50;
    int terminal_top_k = 3;
    bool save_frames = false;
    std::string save_dir = "pi_debug_frames";
    int save_every = 30;
};
