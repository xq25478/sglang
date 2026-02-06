#include <zmq.hpp>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdlib>

namespace py = pybind11;
static auto packb = py::module_::import("msgpack").attr("packb");
static auto unpackb = py::module_::import("msgpack").attr("unpackb");

// =====================================================
// DEBUG 开关（环境变量）
// =====================================================
static inline bool remote_spec_debug_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("REMOTE_SPEC_DEBUG");
        return env && std::string(env) == "1";
    }();
    return enabled;
}

// =====================================================
// CRTP 基类
// =====================================================
template <typename Derived>
class AsyncZmqEndpointCRTP {
public:
    AsyncZmqEndpointCRTP(zmq::context_t& ctx)
        : ctx_(ctx), running_(false) {}

    void start() {
        if (running_) return;
        static_cast<Derived*>(this)->do_bind_or_connect();
        running_ = true;
        recv_thread_ = std::thread(&AsyncZmqEndpointCRTP::recv_loop, this);
        send_thread_ = std::thread(&AsyncZmqEndpointCRTP::send_loop, this);
    }

    void stop() {
        if (!running_) return;
        running_ = false;

        // 1. 先 close socket，打断 poll
        static_cast<Derived*>(this)->sock().close();

        // 2. 唤醒 send 线程
        send_cv_.notify_all();

        // 3. join 线程
        if (recv_thread_.joinable()) recv_thread_.join();
        if (send_thread_.joinable()) send_thread_.join();
    }

protected:
    zmq::context_t& ctx_;
    std::atomic<bool> running_;
    std::thread recv_thread_;
    std::thread send_thread_;

    std::mutex send_cv_mtx_;
    std::condition_variable send_cv_;

private:
    void recv_loop() {
        zmq_pollitem_t items[] = {{
            static_cast<void*>(static_cast<Derived*>(this)->sock()),
            0, ZMQ_POLLIN, 0
        }};

        try {
            while (running_) {
                zmq::poll(items, 1, std::chrono::milliseconds(1));
                if (!running_) break;

                if (items[0].revents & ZMQ_POLLIN) {
                    static_cast<Derived*>(this)->handle_recv();
                }
            }
        } catch (const zmq::error_t& e) {
            // socket close 会触发异常，这是预期行为
            if (running_) {
                std::cerr << "[recv_loop] zmq exception: " << e.what() << std::endl;
            }
        }
    }

    void send_loop() {
        while (running_) {
            std::unique_lock<std::mutex> lock(send_cv_mtx_);
            send_cv_.wait(lock, [this] {
                Derived* self = static_cast<Derived*>(this);
                return self->has_send_data() || !running_;
            });

            if (!running_) break;
            static_cast<Derived*>(this)->handle_send();
        }
    }
};

// =====================================================
// DEALER Endpoint
// =====================================================
class DealerEndpoint : public AsyncZmqEndpointCRTP<DealerEndpoint> {
public:
    DealerEndpoint(
        const std::string& name,
        const std::string& addr,
        bool bind = false)
        : AsyncZmqEndpointCRTP(internal_ctx_),
          sock_(internal_ctx_, zmq::socket_type::dealer),
          name_(name),
          addr_(addr),
          bind_(bind) {

        sock_.set(zmq::sockopt::routing_id, name_);
        sock_.set(zmq::sockopt::sndhwm, 10000);
        sock_.set(zmq::sockopt::rcvhwm, 10000);
        sock_.set(zmq::sockopt::linger, 0);
        sock_.set(zmq::sockopt::immediate, 1);
        sock_.set(zmq::sockopt::sndbuf, 4 * 1024 * 1024);
        sock_.set(zmq::sockopt::rcvbuf, 4 * 1024 * 1024);
    }

    ~DealerEndpoint() {
        stop(); // 析构函数添加 stop，保证 Python 端程序停止，自动调用 stop 函数。
    }

    bool has_send_data() {
        std::lock_guard<std::mutex> lock(send_mtx_);
        return !send_queue_.empty();
    }

    void send_bytes(std::string&& data) {
        {
            std::lock_guard<std::mutex> lock(send_mtx_);
            send_queue_.push(std::move(data));
        }
        send_cv_.notify_one();
    }

    void send_bytes_batch(std::vector<std::string>&& datas) {
        {
            std::lock_guard<std::mutex> lock(send_mtx_);
            for (auto& d : datas) {
                send_queue_.push(std::move(d));
            }
        }
        send_cv_.notify_one();
    }

    std::vector<std::string> get_received_bytes() {
        std::lock_guard<std::mutex> lock(recv_mtx_);
        std::vector<std::string> out;
        out.reserve(recv_queue_.size());
        while (!recv_queue_.empty()) {
            out.push_back(std::move(recv_queue_.front()));
            recv_queue_.pop();
        }
        return out;
    }

    void do_bind_or_connect() {
        if (bind_) sock_.bind(addr_);
        else sock_.connect(addr_);
    }

    zmq::socket_t& sock() { return sock_; }

    void handle_recv() {
        try {
            auto t0 = std::chrono::steady_clock::now();
            size_t recv_cnt = 0;

            std::vector<std::string> batch;
            batch.reserve(64);

            while (true) {
                zmq::message_t frame;
                auto res = sock_.recv(frame, zmq::recv_flags::dontwait);
                if (!res) break;

                std::string data(static_cast<char*>(frame.data()), frame.size());
                while (sock_.get(zmq::sockopt::rcvmore)) {
                    zmq::message_t more;
                    auto res = sock_.recv(more, zmq::recv_flags::none);
                    if (!res) break;
                    data.append(static_cast<char*>(more.data()), more.size());
                }

                batch.push_back(std::move(data));
                ++recv_cnt;
            }

            if (!batch.empty()) {
                std::lock_guard<std::mutex> lock(recv_mtx_);
                for (auto& s : batch) recv_queue_.push(std::move(s));
            }

            if (remote_spec_debug_enabled() && recv_cnt > 0) {
                auto t1 = std::chrono::steady_clock::now();
                auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                std::cerr
                    << "[REMOTE_SPEC_DEBUG C++][RECV] "
                    << "msgs nums=" << recv_cnt
                    << " time_us=" << us
                    << std::endl;
            }

        } catch (const zmq::error_t&) {}
    }

    void handle_send() {
        try {
            auto t0 = std::chrono::steady_clock::now();
            size_t send_cnt = 0;

            std::queue<std::string> tmp;
            {
                std::lock_guard<std::mutex> lock(send_mtx_);
                std::swap(tmp, send_queue_);
            }

            while (!tmp.empty()) {
                sock_.send(zmq::message_t{}, zmq::send_flags::sndmore);
                sock_.send(zmq::buffer(tmp.front()), zmq::send_flags::none);
                tmp.pop();
                ++send_cnt;
            }

            if (remote_spec_debug_enabled() && send_cnt > 0) {
                auto t1 = std::chrono::steady_clock::now();
                auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                std::cerr
                    << "[REMOTE_SPEC_DEBUG C++][SEND] "
                    << "msgs nums=" << send_cnt
                    << " time_us=" << us
                    << std::endl;
            }

        } catch (const zmq::error_t&) {}
    }

private:
    zmq::context_t internal_ctx_{1};
    zmq::socket_t sock_;
    std::string name_, addr_;
    bool bind_;

    std::mutex send_mtx_;
    std::queue<std::string> send_queue_;

    std::mutex recv_mtx_;
    std::queue<std::string> recv_queue_;
};

// =====================================================
// PYBIND11
// =====================================================
PYBIND11_MODULE(remote_spec_zmq, m) {
    py::class_<DealerEndpoint>(m, "DealerEndpoint")
        .def(py::init<const std::string&, const std::string&, bool>(),
             py::arg("name"), py::arg("addr"), py::arg("bind") = false)
        .def("start", &DealerEndpoint::start)
        .def("stop", &DealerEndpoint::stop)
        .def("send_obj", [](DealerEndpoint& self, py::object obj) {
            py::bytes b = packb(obj, py::arg("use_bin_type") = true);
            std::string data = b;
            py::gil_scoped_release release;
            self.send_bytes(std::move(data));
        })
        .def("send_objs", [](DealerEndpoint& self, py::iterable objs) {
            std::vector<std::string> packed_batch;

            for (auto obj : objs) {
                py::bytes b = packb(obj, py::arg("use_bin_type") = true);
                packed_batch.push_back(std::string(b));
            }

            py::gil_scoped_release release;
            self.send_bytes_batch(std::move(packed_batch));
        })
        .def("get_received_objs", [](DealerEndpoint& self) {
            std::vector<std::string> datas;
            {
                py::gil_scoped_release release;
                datas = self.get_received_bytes();
            }
            py::list out;
            for (auto& d : datas)
                out.append(unpackb(py::bytes(d), py::arg("raw") = false));
            return out;
        });
}
