#include "remote_spec_zmq_logging.hpp"

#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

namespace {

class RemoteSpecAsyncLogger {
public:
    static RemoteSpecAsyncLogger& instance() {
        static RemoteSpecAsyncLogger logger;
        return logger;
    }

    void enqueue(std::string msg) {
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            pending_.push_back(std::move(msg));
        }
        queue_cv_.notify_one();
    }

private:
    RemoteSpecAsyncLogger() : worker_(&RemoteSpecAsyncLogger::run, this) {}

    ~RemoteSpecAsyncLogger() {
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            stopping_ = true;
        }
        queue_cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void run() {
        std::deque<std::string> local_queue;
        std::string batch_output;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(queue_mtx_);
                queue_cv_.wait(lock, [this] {
                    return stopping_ || !pending_.empty();
                });
                if (stopping_ && pending_.empty()) {
                    break;
                }
                local_queue.swap(pending_);
            }

            size_t total_bytes = 0;
            for (const auto& msg : local_queue) {
                total_bytes += msg.size() + 1;
            }

            batch_output.clear();
            batch_output.reserve(total_bytes);
            for (auto& msg : local_queue) {
                batch_output.append(msg);
                batch_output.push_back('\n');
            }
            local_queue.clear();

            if (!batch_output.empty()) {
                std::fwrite(batch_output.data(), 1, batch_output.size(), stdout);
                std::fflush(stdout);
            }
        }
    }

    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::deque<std::string> pending_;
    bool stopping_ = false;
    std::thread worker_;
};

}  // namespace

bool remote_spec_debug_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("REMOTE_SPEC_DEBUG");
        return env && std::string(env) == "1";
    }();
    return enabled;
}

void remote_spec_enqueue_log(std::string msg) {
    RemoteSpecAsyncLogger::instance().enqueue(std::move(msg));
}
