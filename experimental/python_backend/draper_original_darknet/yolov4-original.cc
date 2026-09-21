/***********************************************************************
 *  Darknet YOLOv4 Inference Benchmark (enhanced)
 *
 *  New features
 *  -----------------------------------------------------------------
 *  •  --cpu-only        – run the network on the CPU even if a GPU is
 *                         available.
 *  •  --no-instrument   – skip per-iteration timing and system-stats
 *                         collection entirely; useful when you want the
 *                         lowest possible overhead during profiling with
 *                         an external tool (e.g. nsys, perf).
 *  •  --help            – prints a short usage message.
 *  •  All parameters (iterations, image file, detection threshold) are
 *     parsed with getopt_long so they can be supplied in any order.
 *  •  Timers around the warm-up run and around *all* inference runs.
 *  •  Per-iteration timings are still collected and printed (unless
 *     --no-instrument is set).
 *  •  CPU statistics: number of logical cores, wall-time, CPU-time and
 *     % utilisation (wall-time * cores).  Measured only over the
 *     inference window (rusage snapshot before/after the loop).
 *  •  GPU statistics (when a GPU is used):
 *        – average utilisation sampled on a background thread during
 *          the actual inference loop (fixes the "always low" bug)
 *        – memory used at the end of the benchmark
 *  •  All statistics are printed together at the end of the program.
 *
 *  Build (example):
 *      g++ -O2 -std=c++17 -I/usr/include/nvidia-ml \
 *          -I/path/to/darknet/include benchmark.cpp -ldarknet -lcuda \
 *          -lnvidia-ml -lpthread -lrt -lstdc++
 *
 *  The program still expects the Darknet repository to be at
 *  /home/dockeruser/darknet (or you may change that constant below).
 ***********************************************************************/

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <cstring>
#include <unistd.h>
#include <getopt.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#ifndef NO_GPU_MONITOR
#include <nvml.h>
#endif

extern "C" {
    #include "darknet.h"

    // Additional Darknet function declarations
    int gpu_index;           // Darknet's global GPU index
    void cuda_set_device(int n);  // Darknet's CUDA device selector
}

/* --------------------------------------------------------------------
 *  Simple high-resolution timer
 * -------------------------------------------------------------------- */
class Timer {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    void stop()  { end_time   = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   end_time - start_time).count() / 1000.0;
    }
};

/* --------------------------------------------------------------------
 *  Statistics helpers
 * -------------------------------------------------------------------- */
void print_inference_stats(const std::vector<double>& times,
                           double total_ms,
                           double warmup_ms) {
    if (times.empty()) return;

    double sum = 0.0, min_t = times[0], max_t = times[0];
    for (double v : times) {
        sum += v;
        if (v < min_t) min_t = v;
        if (v > max_t) max_t = v;
    }
    double avg = sum / times.size();

    std::cout << "\n=== Inference Statistics ===\n";
    std::cout << "Warm-up time     : " << std::fixed << std::setprecision(2)
              << warmup_ms << " ms\n";
    std::cout << "Iterations       : " << times.size() << "\n";
    std::cout << "Total inference  : " << total_ms << " ms\n";
    std::cout << "Average per iter : " << avg << " ms ("
              << 1000.0 / avg << " FPS)\n";
    std::cout << "Min / Max        : " << min_t << " ms / " << max_t << " ms\n";
}

/* --------------------------------------------------------------------
 *  CPU usage – snapshot rusage before and after the inference window
 *  so load time does not skew the numbers.
 * -------------------------------------------------------------------- */
struct RUsageSnapshot {
    long user_us   = 0;
    long system_us = 0;
};

RUsageSnapshot capture_rusage() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return {
        ru.ru_utime.tv_sec * 1'000'000L + ru.ru_utime.tv_usec,
        ru.ru_stime.tv_sec * 1'000'000L + ru.ru_stime.tv_usec
    };
}

struct CpuStat {
    double cpu_time_sec   = 0.0;   // user + system, inference window only
    double wall_time_sec  = 0.0;
    double util_percent   = 0.0;
    int    logical_cores  = 1;
};

CpuStat compute_cpu_stat(const RUsageSnapshot& before,
                          const RUsageSnapshot& after,
                          double wall_ms) {
    CpuStat s;
    s.logical_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (s.logical_cores == 0) s.logical_cores = 1;

    long delta_us = (after.user_us   - before.user_us) +
                    (after.system_us - before.system_us);
    s.cpu_time_sec  = delta_us / 1'000'000.0;
    s.wall_time_sec = wall_ms / 1000.0;

    if (s.wall_time_sec > 0.0)
        s.util_percent = (s.cpu_time_sec / s.wall_time_sec)
                         * 100.0 / s.logical_cores;
    return s;
}

#ifndef NO_GPU_MONITOR
/* --------------------------------------------------------------------
 *  GPU utilisation – polled on a background thread *during* inference
 *  so the samples reflect actual load rather than the idle period
 *  after the benchmark finishes.
 * -------------------------------------------------------------------- */
struct GpuStat {
    bool     available  = false;
    unsigned index      = 0;
    double   avg_util   = 0.0;   // [%] averaged over all samples
    size_t   mem_used   = 0;     // bytes, read once after the benchmark
};

class GpuPoller {
public:
    explicit GpuPoller(unsigned device_index = 0,
                       unsigned poll_ms = 100)
        : device_index_(device_index), poll_ms_(poll_ms) {}

    // Call before starting the inference loop.
    bool init() {
        if (nvmlInit() != NVML_SUCCESS) return false;
        unsigned devCount = 0;
        if (nvmlDeviceGetCount(&devCount) != NVML_SUCCESS || devCount == 0) {
            nvmlShutdown(); return false;
        }
        if (nvmlDeviceGetHandleByIndex(device_index_, &dev_) != NVML_SUCCESS) {
            nvmlShutdown(); return false;
        }
        available_ = true;
        return true;
    }

    void start_polling() {
        if (!available_) return;
        stop_flag_.store(false);
        thread_ = std::thread([this]{ poll_loop(); });
    }

    void stop_polling() {
        stop_flag_.store(true);
        if (thread_.joinable()) thread_.join();
    }

    // Call after stop_polling() to read memory.
    GpuStat finalize() {
        GpuStat gs;
        if (!available_) return gs;

        gs.available = true;
        gs.index     = device_index_;

        {
            std::lock_guard<std::mutex> lk(mu_);
            gs.avg_util = sample_count_ > 0
                          ? util_sum_ / sample_count_
                          : 0.0;
        }

        nvmlMemory_t mem;
        if (nvmlDeviceGetMemoryInfo(dev_, &mem) == NVML_SUCCESS)
            gs.mem_used = static_cast<size_t>(mem.used);

        nvmlShutdown();
        return gs;
    }

private:
    void poll_loop() {
        while (!stop_flag_.load()) {
            nvmlUtilization_t util;
            if (nvmlDeviceGetUtilizationRates(dev_, &util) == NVML_SUCCESS) {
                std::lock_guard<std::mutex> lk(mu_);
                util_sum_ += util.gpu;
                ++sample_count_;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(poll_ms_));
        }
    }

    unsigned      device_index_;
    unsigned      poll_ms_;
    bool          available_    = false;
    nvmlDevice_t  dev_          = {};
    std::thread   thread_;
    std::atomic<bool> stop_flag_{false};

    std::mutex    mu_;
    double        util_sum_     = 0.0;
    unsigned long sample_count_ = 0;
};
#else
/* --------------------------------------------------------------------
 *  Stub GPU monitoring when NO_GPU_MONITOR is defined (CPU-only build)
 * -------------------------------------------------------------------- */
struct GpuStat {
    bool     available  = false;
    unsigned index      = 0;
    double   avg_util   = 0.0;
    size_t   mem_used   = 0;
};

class GpuPoller {
public:
    explicit GpuPoller(unsigned = 0, unsigned = 0) {}
    bool init() { return false; }
    void start_polling() {}
    void stop_polling() {}
    GpuStat finalize() { return GpuStat{}; }
};
#endif  // NO_GPU_MONITOR

/* --------------------------------------------------------------------
 *  Help / usage
 * -------------------------------------------------------------------- */
static const char* const usage_msg =
R"(Usage: benchmark [options]

Options:
  -i, --iterations N      Number of inference iterations (default 1000)
  -f, --file PATH         Image file to run inference on (default data/dog.jpg)
  -t, --threshold VAL     Detection threshold (default 0.25)
  -n, --no-instrument     Skip per-iteration timing and system statistics;
                          useful when profiling with an external tool
                          (e.g. nsys, perf) where instrumentation overhead
                          would distort results
  -h, --help              Show this help message and exit

All arguments are optional and can appear in any order.

Note: This benchmark is available in two versions:
  - yolov4       : GPU inference with GPU utilization monitoring
  - yolov4_cpu   : CPU-only inference (requires libdarknet_cpu.so)

Example:
    ./yolov4 --iterations 500 --file my.jpg --threshold 0.30
    ./yolov4_cpu --no-instrument --iterations 2000
)";

void print_help_and_exit() {
    std::cout << usage_msg << std::endl;
    std::exit(EXIT_SUCCESS);
}

/* --------------------------------------------------------------------
 *  Main
 * -------------------------------------------------------------------- */
int main(int argc, char **argv) {
    /* ---- 1. Change working directory to the Darknet repo --------------- */
    const char* darknet_dir = "/home/dockeruser/sei-wohlbier-darknet";
    if (chdir(darknet_dir) != 0) {
        std::cerr << "Error: Could not change to directory: " << darknet_dir
                  << "\nPlease update darknet_dir in the source code or run "
                     "from the darknet directory.\n";
        return 1;
    }

    /* ---- 2. Default configuration -------------------------------------- */
    const char* datacfg    = "cfg/coco.data";
    const char* cfgfile    = "cfg/yolov4.cfg";
    const char* weightfile = "yolov4.weights";

    std::string filename   = "data/dog.jpg";  // std::string avoids dangling optarg ptr
    int         iterations = 1000;
    float       thresh     = 0.25f;
    bool        no_instrument  = false;

    /* ---- 3. Parse command-line arguments ------------------------------- */
    static struct option long_opts[] = {
        {"iterations",    required_argument, nullptr, 'i'},
        {"file",          required_argument, nullptr, 'f'},
        {"threshold",     required_argument, nullptr, 't'},
        {"no-instrument", no_argument,       nullptr, 'n'},
        {"help",          no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "i:f:t:nh", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'i': iterations    = std::stoi(optarg);  break;
            case 'f': filename      = optarg;             break;
            case 't': thresh        = std::stof(optarg);  break;
            case 'n': no_instrument = true;               break;
            case 'h': print_help_and_exit();              break;
            default:
                std::cerr << "Unknown option. Use --help.\n";
                return 1;
        }
    }

    /* ---- 4. Print configuration ---------------------------------------- */
    std::cout << "Darknet YOLOv4 Inference Benchmark\n"
              << "===================================\n"
              << "Working directory : " << darknet_dir    << '\n'
              << "Config            : " << cfgfile        << '\n'
              << "Weights           : " << weightfile     << '\n'
              << "Image             : " << filename       << '\n'
              << "Iterations        : " << iterations     << '\n'
              << "Threshold         : " << thresh         << '\n'
              << "No-instrument?    : " << (no_instrument  ? "yes" : "no") << "\n\n";

    /* ---- 5. Load the network ------------------------------------------- */
    Timer timer;
    std::cout << "Loading network..." << std::endl;
    timer.start();
    network* net = load_network_custom(
        const_cast<char*>(cfgfile),
        const_cast<char*>(weightfile), 1, 1);
    timer.stop();
    std::cout << "Network loaded in " << std::fixed << std::setprecision(2)
              << timer.elapsed_ms() << " ms\n";

    /* ---- 7. Load the image --------------------------------------------- */
    std::cout << "Loading image..." << std::endl;
    timer.start();
    image im = load_image_color(const_cast<char*>(filename.c_str()), 0, 0);
    timer.stop();
    std::cout << "Image loaded in "  << timer.elapsed_ms() << " ms\n"
              << "Image size: " << im.w << "x" << im.h << "\n";

    /* ---- 8. Prepare input (letterbox) ---------------------------------- */
    image  sized = letterbox_image(im, net->w, net->h);
    float* X     = sized.data;

    /* ---- 9. Warm-up run ------------------------------------------------ */
    Timer warmup_timer;
    warmup_timer.start();
    network_predict(*net, X);
    warmup_timer.stop();
    std::cout << "\nWarm-up completed in "
              << warmup_timer.elapsed_ms() << " ms\n";

    /* ---- 10. Inference loop -------------------------------------------- */
    std::vector<double> iter_times;

    // Only allocate the per-iteration timing vector if instrumentation is on.
    if (!no_instrument)
        iter_times.reserve(iterations);

    // Initialise the GPU poller BEFORE the loop so polling starts promptly.
    // For CPU-only builds (NO_GPU_MONITOR defined), the poller is a stub.
    GpuPoller gpu_poller(/*device_index=*/0, /*poll_ms=*/100);
#ifndef NO_GPU_MONITOR
    if (!no_instrument)
        gpu_poller.init();
#endif

    // Capture rusage snapshot just before the loop starts.
    RUsageSnapshot rusage_before = capture_rusage();

    Timer total_inf_timer;
    total_inf_timer.start();

#ifndef NO_GPU_MONITOR
    if (!no_instrument)
        gpu_poller.start_polling();
#endif

    for (int i = 0; i < iterations; ++i) {
        if (!no_instrument) {
            Timer it;
            it.start();
            network_predict(*net, X);
            it.stop();

            double elapsed = it.elapsed_ms();
            iter_times.push_back(elapsed);

            if ((i + 1) % 10 == 0) {
                std::cout << "Iteration " << (i + 1) << "/" << iterations
                          << " – last: " << elapsed << " ms\n";
            }
        } else {
            // No overhead: raw inference call only.
            network_predict(*net, X);
        }
    }

    total_inf_timer.stop();
    const double total_inf_ms = total_inf_timer.elapsed_ms();

#ifndef NO_GPU_MONITOR
    if (!no_instrument)
        gpu_poller.stop_polling();
#endif

    // Capture rusage snapshot immediately after the loop ends.
    RUsageSnapshot rusage_after = capture_rusage();

    /* ---- 11. Compute and print statistics ------------------------------ */
    if (!no_instrument) {
        print_inference_stats(iter_times, total_inf_ms,
                              warmup_timer.elapsed_ms());

        CpuStat cpu = compute_cpu_stat(rusage_before, rusage_after,
                                       total_inf_ms);
#ifndef NO_GPU_MONITOR
        GpuStat gpu = gpu_poller.finalize();
#else
        GpuStat gpu = GpuStat{};
#endif

        std::cout << "\n=== System Statistics ===\n";
        std::cout << "Logical CPU cores          : " << cpu.logical_cores << '\n';
        std::cout << "CPU time (user+sys)        : " << std::fixed
                  << std::setprecision(2) << cpu.cpu_time_sec << " s\n";
        std::cout << "Wall-time (inference only) : " << cpu.wall_time_sec << " s\n";
        std::cout << "Average CPU utilisation    : " << cpu.util_percent  << " %\n";

        if (gpu.available) {
            std::cout << "GPU [" << gpu.index << "] avg utilisation  : "
                      << std::setprecision(2) << gpu.avg_util << " %\n";
            std::cout << "GPU memory used (post-run) : "
                      << (gpu.mem_used / (1024.0 * 1024.0)) << " MiB\n";
        }
#ifdef NO_GPU_MONITOR
        else {
            std::cout << "GPU statistics             : N/A (CPU-only build)\n";
        }
#else
        else {
            std::cout << "GPU statistics             : not available "
                         "(no GPU or NVML missing)\n";
        }
#endif
    } else {
        // Minimal output when instrumentation is disabled.
        std::cout << "\nTotal wall-time (inference only): "
                  << std::fixed << std::setprecision(2)
                  << total_inf_ms << " ms ("
                  << (1000.0 * iterations / total_inf_ms) << " FPS)\n"
                  << "(per-iteration stats and system statistics skipped "
                     "due to --no-instrument)\n";
    }

    /* ---- 12. Run detection once and show results ----------------------- */
    std::cout << "\n=== Detection Results (final pass) ===\n";
    int nboxes = 0;
    constexpr float nms_thresh = 0.45f;
    detection* dets = get_network_boxes(net, im.w, im.h,
                                         thresh, 0.5f, 0, 1, &nboxes, 1);
    layer l = net->layers[net->n - 1];
    if (nms_thresh > 0.0f)
        do_nms_sort(dets, nboxes, l.classes, nms_thresh);

    metadata meta = get_metadata(const_cast<char*>(datacfg));
    int detected_objs = 0;
    for (int i = 0; i < nboxes; ++i) {
        for (int j = 0; j < l.classes; ++j) {
            if (dets[i].prob[j] > thresh) {
                std::cout << meta.names[j] << ": "
                          << std::fixed << std::setprecision(2)
                          << (dets[i].prob[j] * 100.0) << " %\n";
                ++detected_objs;
            }
        }
    }
    std::cout << "Total detections: " << detected_objs << "\n";

    /* ---- 13. Cleanup --------------------------------------------------- */
    free_detections(dets, nboxes);
    free_image(im);
    free_image(sized);
    free_network(*net);

    std::cout << "\n=== Benchmark finished ===\n";
    return 0;
}
