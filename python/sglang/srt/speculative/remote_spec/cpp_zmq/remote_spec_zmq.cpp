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
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <tuple>
#include <msgpack.hpp>

#include <remote_spec_protocol.hpp>

using namespace std;
using namespace remote_spec;
namespace py = pybind11;

// 设置 export REMOTE_SPEC_DEBUG=1 代码会打印相关关键信息
static inline bool remote_spec_debug_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("REMOTE_SPEC_DEBUG");
        return env && std::string(env) == "1";
    }();
    return enabled;
}

// =====================================================
// Msgpack 自定义序列化流：直接写入 std::string 避免二次拷贝
// =====================================================
struct StringStream {
    std::string& str;
    StringStream(std::string& s) : str(s) {}
    void write(const char* buf, size_t len) {
        str.append(buf, len);
    }
};

// =====================================================
// 优化后的无锁 SafeQueue (基于 RingBuffer 实现)
// =====================================================
template <typename T>
class SafeQueue {
public:
    // 使用 2 的幂次方大小以便进行位运算优化
    explicit SafeQueue(size_t capacity = 65536) 
        : capacity_(capacity), mask_(capacity - 1), head_(0), tail_(0) {
        // 确保 capacity 是 2 的幂
        if ((capacity & (capacity - 1)) != 0) {
            capacity_ = 65536; 
            mask_ = capacity_ - 1;
        }
        buffer_.resize(capacity_);
    }
    void push(T&& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) & mask_;
        
        // 如果队列满了，这里采用阻塞/等待策略以维持 SafeQueue 的原有行为
        while (next_tail == head_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        
        buffer_[current_tail] = std::move(item);
        tail_.store(next_tail, std::memory_order_release);
        
        // 唤醒可能正在等待的 pop (兼容原有 CV 逻辑)
        {
            std::lock_guard<std::mutex> lock(cv_mtx_);
            cv_.notify_one();
        }
    }
    bool pop(T& item, int timeout_ms = 1) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        
        if (current_head == tail_.load(std::memory_order_acquire)) {
            // 队列为空，使用条件变量等待以节省 CPU
            std::unique_lock<std::mutex> lock(cv_mtx_);
            if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
                return head_.load(std::memory_order_relaxed) != tail_.load(std::memory_order_acquire);
            })) {
                return false;
            }
            current_head = head_.load(std::memory_order_relaxed);
        }
        item = std::move(buffer_[current_head]);
        head_.store((current_head + 1) & mask_, std::memory_order_release);
        return true;
    }
    // 辅助接口：唤醒所有阻塞在 pop 的线程，用于 stop()
    void notify_all() {
        std::lock_guard<std::mutex> lock(cv_mtx_);
        cv_.notify_all();
    }
    // 辅助检查头部元素是否超时 (T 需包含 timestamp)
    bool peek_and_check_timeout(long long timeout_ms) {
        size_t current_head = head_.load(std::memory_order_acquire);
        if (current_head == tail_.load(std::memory_order_acquire)) return false;
        auto now = std::chrono::steady_clock::now();
        auto msg_time = buffer_[current_head].timestamp;
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - msg_time).count();
        
        return diff > timeout_ms;
    }
    vector<T> pop_all() {
        vector<T> res;
        T item;
        // 尽力而为地排空当前所有可用数据
        while (head_.load(std::memory_order_relaxed) != tail_.load(std::memory_order_acquire)) {
            size_t current_head = head_.load(std::memory_order_relaxed);
            res.push_back(std::move(buffer_[current_head]));
            head_.store((current_head + 1) & mask_, std::memory_order_release);
        }
        return res;
    }
    size_t size() {
        size_t h = head_.load(std::memory_order_acquire);
        size_t t = tail_.load(std::memory_order_acquire);
        if (t >= h) return t - h;
        return capacity_ - (h - t);
    }
    void clear() {
        T dummy;
        while (pop(dummy, 0));
    }
private:
    size_t capacity_;
    size_t mask_;
    std::vector<T> buffer_;
    
    // 使用原子操作实现无锁索引
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
    
    // 为了兼容原有代码的阻塞等待模式，保留辅助的 CV
    std::mutex cv_mtx_;
    std::condition_variable cv_;
};

// =====================================================
// 封装带时间戳的接收对象
// =====================================================
template<typename DataT>
struct TimestampedObj {
    DataT data;
    std::chrono::steady_clock::time_point timestamp;
};

// =====================================================
// 核心基类：双线程异步架构
// =====================================================
template <typename Derived, typename SendT, typename RecvT, typename RawRecvT = string, typename RawSendT = string>
class [[gnu::visibility("default")]] AsyncZmqEndpointCRTP {
public:
    // 将心跳信号定义为类内常量
    static constexpr const char* DEALER_HEARTBEAT = "DEALER_HEARTBEAT";
    // 监控检查频率. Dealer端 发送心跳信号间隔
    static constexpr int ENDPOINT_MONITOR_INTERNAL_MS = 1000; 
    // 超时阈值：Dealer端 发送心跳信号超时时间
    static constexpr int DEALER_HEARTBEAT_TIMEOUT_MS = 3000; 
    // Python 侧准备就绪队列的超时阈值 (N ms)
    static constexpr int PYTHON_READY_QUEUE_TIMEOUT_MS = 30000; 

    AsyncZmqEndpointCRTP(const string& addr, bool bind, zmq::socket_type sock_type)
        : internal_ctx_(1), sock_(internal_ctx_, sock_type), addr_(addr), bind_(bind), running_(false), sock_type_tag(sock_type) {
        if (bind_){
            endpoint_type = "Router";
        }
        else{
            endpoint_type = "Dealer";
        }
    }
    virtual ~AsyncZmqEndpointCRTP() { stop(); }

    void start() {
        if (running_) return;
        static_cast<Derived*>(this)->do_setup_socket();
        static_cast<Derived*>(this)->do_bind_or_connect();
        running_ = true;
        // 线程一：纯网络 IO 线程，负责 zmq 的收发。
        zmq_io_thread_ = thread(&AsyncZmqEndpointCRTP::zmq_io_loop, this);

        // 线程二：解码处理线程 (Recv Unpack) 负责将 zmq 收到的数据进行解压
        zmq_data_unpack_thread_ = thread(&AsyncZmqEndpointCRTP::zmq_data_unpack_loop, this);

        // 线程三：编码处理线程 (Send Pack) 负责将发送的的数据进行压缩
        zmq_data_pack_thread_ = thread(&AsyncZmqEndpointCRTP::zmq_data_pack_loop, this);

        // 线程四：监控线程 处理 接收队列限流/ Draft 心跳状态监控
        endpoint_monitor_thread_ = thread(&AsyncZmqEndpointCRTP::endpoint_monitor_loop, this);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        cout << "[ZMQ " << endpoint_type << " Start!] addr:" << addr_ << endl;
    }

    void stop() {
        bool expected = true;
        // 1. 先关闭 running_ 标志位
        if (!running_.compare_exchange_strong(expected, false)) return;
        
        try {
            // 2. 强制唤醒所有可能在 pop 等待中的线程，防止 join 死锁
            recv_raw_queue_.notify_all();
            python_ready_queue_.notify_all();
            python_send_queue_.notify_all();
            send_raw_queue_.notify_all();
            internal_ctx_.shutdown();
            if (zmq_io_thread_.joinable()) zmq_io_thread_.join();
            if (zmq_data_unpack_thread_.joinable()) zmq_data_unpack_thread_.join();
            if (zmq_data_pack_thread_.joinable()) zmq_data_pack_thread_.join();
            if (endpoint_monitor_thread_.joinable()) endpoint_monitor_thread_.join();
            sock_.close();
            internal_ctx_.close();
        } catch (...) {}
        cout << "[ZMQ " << endpoint_type << " Stop!] addr:" << addr_ << endl;
    }

    // Python 接口：获取已解包的数据
    vector<RecvT> get_received_objs_internal() {
        auto wrapped_list = python_ready_queue_.pop_all();
        vector<RecvT> out;
        out.reserve(wrapped_list.size());
        for (auto& w : wrapped_list) out.push_back(std::move(w.data));
        return out;
    }

    // Python 接口：发送请求
    void enqueue_send_obj(SendT&& obj) {
        python_send_queue_.push(move(obj));
    }

    void do_bind_or_connect() {
        if (bind_) sock_.bind(addr_);
        else sock_.connect(addr_);
    }

protected:
    void setup_common_socket_options() {
        sock_.set(zmq::sockopt::sndhwm, 10000);
        sock_.set(zmq::sockopt::rcvhwm, 10000);
        sock_.set(zmq::sockopt::linger, 0);
        sock_.set(zmq::sockopt::immediate, 1);
        sock_.set(zmq::sockopt::sndbuf, 10 * 1024 * 1024);
        sock_.set(zmq::sockopt::rcvbuf, 10 * 1024 * 1024);
    }

    // 线程一：IO 循环
    void zmq_io_loop() {
        zmq_pollitem_t items[] = {{ static_cast<void*>(sock_), 0, ZMQ_POLLIN, 0 }};
        while (running_) {
            try {
                RawSendT to_send;
                while (send_raw_queue_.pop(to_send, 0)) {
                    static_cast<Derived*>(this)->handle_raw_send(to_send);
                }
                int rc = zmq_poll(items, 1, 0); 
                if (rc > 0 && (items[0].revents & ZMQ_POLLIN)) {
                    static_cast<Derived*>(this)->handle_raw_recv();
                }
                if (rc == 0 && send_raw_queue_.size() == 0) {
                    // std::this_thread::yield(); 
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } catch (const zmq::error_t& e) {
                if (e.num() == ETERM || !running_) break;
                handle_error(e);
            }
        }
    }

    // 线程二：解码处理线程 (Recv Unpack)
    void zmq_data_unpack_loop() {
        const size_t batch_size = 32;
        vector<RawRecvT> batch_raw;
        batch_raw.reserve(batch_size);
        while (running_) {
            RawRecvT raw_data;
            if (recv_raw_queue_.pop(raw_data, 1)) {
                batch_raw.push_back(move(raw_data));
            }
            
            // 当达到批处理大小或队列暂空时，处理数据
            if (!batch_raw.empty() && (batch_raw.size() >= batch_size || recv_raw_queue_.size() == 0)) {
                for (auto& item : batch_raw) {
                    static_cast<Derived*>(this)->process_incoming_data(move(item));
                }
                batch_raw.clear();
            }
        }
    }

    // 线程三：编码处理线程 (Send Pack)
    void zmq_data_pack_loop() {
        while (running_) {
            SendT req;
            if (python_send_queue_.pop(req, 1)) {
                static_cast<Derived*>(this)->process_outgoing_data(move(req));
            }
        }
    }

    void endpoint_monitor_loop() {
        while (running_) {
            if (python_ready_queue_.peek_and_check_timeout(PYTHON_READY_QUEUE_TIMEOUT_MS)) {
                while (python_ready_queue_.peek_and_check_timeout(PYTHON_READY_QUEUE_TIMEOUT_MS)) {
                    TimestampedObj<RecvT> discarded;
                    if (python_ready_queue_.pop(discarded, 0)) {
                        if (remote_spec_debug_enabled()) {
                            cout << "[Monitor] " << endpoint_type << " 接收数据超时未被处理，已丢弃！" << endl;
                        }
                    }
                }
            }
            static_cast<Derived*>(this)->on_monitor_tick();
            std::this_thread::sleep_for(std::chrono::milliseconds(ENDPOINT_MONITOR_INTERNAL_MS));
        }
    }

    // 默认的 Multipart 接收逻辑
    void handle_raw_recv_base() {
        int max_batch = 100; 
        while (max_batch-- > 0) {
            vector<zmq::message_t> frames;
            while (true) {
                zmq::message_t msg;
                auto res = this->sock_.recv(msg, zmq::recv_flags::dontwait);
                if (!res) return;
                frames.push_back(std::move(msg));
                if (!this->sock_.get(zmq::sockopt::rcvmore)) break;
            }
            if (!frames.empty()) {
                static_cast<Derived*>(this)->dispatch_frames(std::move(frames));
            }
        }
    }

    void handle_error(const zmq::error_t& e) {
        if (running_) {
            cout << "[ZMQ Error] " << addr_ << ": " << e.what() << endl;
            auto err = e.num();
            if (err == ECONNRESET || err == ECONNREFUSED || err == ENETDOWN || 
                err == ENETUNREACH || err == EHOSTUNREACH || err == ETIMEDOUT) {
                reconnect();
            }
        }
    }

    void reconnect() {
        cout << "[ZMQ Reconnecting] " << addr_ << endl;
        sock_.close();
        sock_ = zmq::socket_t(internal_ctx_, sock_type_tag);
        static_cast<Derived*>(this)->do_setup_socket();
        do_bind_or_connect();
        static_cast<Derived*>(this)->on_reconnect();
    }

    string endpoint_type;
    zmq::context_t internal_ctx_;
    zmq::socket_t sock_;
    string addr_;
    bool bind_;
    atomic<bool> running_;
    zmq::socket_type sock_type_tag;

    thread zmq_io_thread_;
    thread zmq_data_unpack_thread_;
    thread zmq_data_pack_thread_;
    thread endpoint_monitor_thread_;

    // 队列链路
    SafeQueue<RawRecvT> recv_raw_queue_;      // IO -> Codec
    SafeQueue<TimestampedObj<RecvT>> python_ready_queue_;   // Codec -> Python
    SafeQueue<SendT> python_send_queue_;    // Python -> Codec
    SafeQueue<RawSendT> send_raw_queue_;    // Codec -> IO
};

// =====================================================
// DEALER Endpoint
// =====================================================
class [[gnu::visibility("default")]] DealerEndpoint : public AsyncZmqEndpointCRTP<DealerEndpoint, RemoteSpecRequest, RemoteSpecRequest, string, string> {
public:
    DealerEndpoint(const string& addr, const string& identity, bool bind = false)
        : AsyncZmqEndpointCRTP(addr, bind, zmq::socket_type::dealer), identity_(identity) {}

    void do_setup_socket() {
        this->sock_.set(zmq::sockopt::routing_id, identity_);
        this->setup_common_socket_options();
    }

    void on_reconnect() { send_heartbeat(); }

    void send_heartbeat() {
        this->send_raw_queue_.push(string(DEALER_HEARTBEAT));
    }

    void on_monitor_tick() {
        send_heartbeat();
        if (remote_spec_debug_enabled()){
            cout << "[Dealer] " << identity_ << " 发送心跳信号到 " << addr_ << endl;  
        }
    }

    void handle_raw_recv() { this->handle_raw_recv_base(); }

    void dispatch_frames(vector<zmq::message_t>&& frames) {
        // Dealer 通常只关心最后一帧作为 payload
        auto& payload = frames.back();
        recv_raw_queue_.push(string(static_cast<char*>(payload.data()), payload.size()));
    }

    void handle_raw_send(string& data) {
        this->sock_.send(zmq::buffer(data), zmq::send_flags::none);
    }

    void process_incoming_data(string&& raw) {
        try {
            msgpack::object_handle oh = msgpack::unpack(raw.data(), raw.size());
            RemoteSpecRequest req = oh.get().as<RemoteSpecRequest>();
            python_ready_queue_.push({std::move(req), chrono::steady_clock::now()});
        } catch (const std::exception& e) {
            if (remote_spec_debug_enabled()) cerr << "[Dealer Unpack Error]: " << e.what() << endl;
        }
    }

    void process_outgoing_data(RemoteSpecRequest&& req) {
        string raw_data;
        // 预分配内存，降低频繁申请造成的碎片和开销
        raw_data.reserve(512); 
        StringStream ss(raw_data);
        msgpack::pack(ss, req);
        send_raw_queue_.push(std::move(raw_data));
    }

private:
    string identity_;
};

// =====================================================
// ROUTER Endpoint
// =====================================================
class [[gnu::visibility("default")]] RouterEndpoint : public AsyncZmqEndpointCRTP<RouterEndpoint, pair<string, RemoteSpecRequest>, pair<string, RemoteSpecRequest>, tuple<string, string>, pair<string, string>> {
public:
    RouterEndpoint(const string& addr, bool bind = true)
        : AsyncZmqEndpointCRTP(addr, bind, zmq::socket_type::router) {}

    void do_setup_socket() { this->setup_common_socket_options(); }
    void on_reconnect() {}

    void on_monitor_tick() {
        auto now = chrono::steady_clock::now();
        lock_guard<mutex> lk(reg_mtx_);
        auto it = registered_dealers_.begin();
        while (it != registered_dealers_.end()) {
            if (chrono::duration_cast<chrono::milliseconds>(now - it->second).count() > DEALER_HEARTBEAT_TIMEOUT_MS) {
                if (remote_spec_debug_enabled()) cout << "[Router] " << it->first << " 心跳信号超时，剔除!" << endl;
                it = registered_dealers_.erase(it);
            } else ++it;
        }
    }

    void handle_raw_recv() { this->handle_raw_recv_base(); }

    void dispatch_frames(vector<zmq::message_t>&& frames) {
        if (frames.size() < 2) return;
        string id(static_cast<char*>(frames[0].data()), frames[0].size());
        string payload(static_cast<char*>(frames.back().data()), frames.back().size());
        
        if (payload == DEALER_HEARTBEAT) {
            lock_guard<mutex> lk(reg_mtx_);
            registered_dealers_[id] = chrono::steady_clock::now();
            if (remote_spec_debug_enabled()) cout << "[Router] " << addr_ << " 收到心跳信号，来自 " << id << endl;
        } else {
            recv_raw_queue_.push(make_tuple(move(id), move(payload)));
        }
    }

    void handle_raw_send(pair<string, string>& data) {
        this->sock_.send(zmq::buffer(data.first), zmq::send_flags::sndmore);
        this->sock_.send(zmq::message_t{}, zmq::send_flags::sndmore); 
        this->sock_.send(zmq::buffer(data.second), zmq::send_flags::none);
    }

    void process_incoming_data(tuple<string, string>&& raw) {
        try {
            string id = get<0>(raw);
            msgpack::object_handle oh = msgpack::unpack(get<1>(raw).data(), get<1>(raw).size());
            RemoteSpecRequest req = oh.get().as<RemoteSpecRequest>();
            python_ready_queue_.push({{move(id), move(req)}, chrono::steady_clock::now()});
        } catch (const std::exception& e) {
            if (remote_spec_debug_enabled()) cerr << "[Router Unpack Error]: " << e.what() << endl;
        }
    }

    void process_outgoing_data(pair<string, RemoteSpecRequest>&& req) {
        string raw_data;
        raw_data.reserve(512); 
        StringStream ss(raw_data);
        msgpack::pack(ss, req.second);
        send_raw_queue_.push({move(req.first), move(raw_data)});
    }

    vector<string> get_all_dealers() {
        lock_guard<mutex> lock(reg_mtx_);
        vector<string> res;
        for (auto& p : registered_dealers_) res.push_back(p.first);
        return res;
    }

private:
    mutex reg_mtx_;
    unordered_map<string, chrono::steady_clock::time_point> registered_dealers_;
};

// =====================================================
// PYBIND11
// =====================================================
PYBIND11_MODULE(remote_spec_zmq, m) {
    py::class_<DealerEndpoint>(m, "DealerEndpoint")
        .def(py::init<const string&, const string&, bool>())
        .def("start", &DealerEndpoint::start)
        .def("stop", &DealerEndpoint::stop)
        .def("send_obj", [](DealerEndpoint& self, py::dict obj) {
            self.enqueue_send_obj(from_py_dict(obj)); 
        })
        .def("send_objs", [](DealerEndpoint& self, py::list objs) {
            for (auto item : objs) self.enqueue_send_obj(from_py_dict(item.cast<py::dict>()));
        })
        .def("get_received_objs", [](DealerEndpoint& self) {
            py::list out;
            auto datas = self.get_received_objs_internal();
            for (auto& obj : datas) out.append(to_py_dict(obj));
            return out;
        });

    py::class_<RouterEndpoint>(m, "RouterEndpoint")
        .def(py::init<const string&, bool>())
        .def("start", &RouterEndpoint::start)
        .def("stop", &RouterEndpoint::stop)
        .def("get_all_dealers", &RouterEndpoint::get_all_dealers)
        .def("send_obj", [](RouterEndpoint& self, const string& id, py::dict obj) {
            self.enqueue_send_obj({id, from_py_dict(obj)});
        })
        .def("send_objs", [](RouterEndpoint& self, const string& id, py::list objs) {
            for (auto item : objs) self.enqueue_send_obj({id, from_py_dict(item.cast<py::dict>())});
        })
        .def("get_received_objs", [](RouterEndpoint& self) {
            py::list out;
            auto datas = self.get_received_objs_internal();
            for (auto& p : datas) out.append(py::make_tuple(p.first, to_py_dict(p.second)));
            return out;
        });
}