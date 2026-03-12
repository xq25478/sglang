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
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <memory>
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

template <typename... Args>
static void remote_spec_debug_log(const Args&... args) {
    if (!remote_spec_debug_enabled()) return;
    static std::mutex log_mtx;
    std::lock_guard<std::mutex> lock(log_mtx);
    (std::cout << ... << args) << std::endl;
}

static inline long long remote_spec_duration_us(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static inline double remote_spec_now_us() {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

static void stamp_request_batch(
    std::vector<RemoteSpecRequest>& reqs,
    double RemoteSpecRequest::* field,
    double timestamp_us = remote_spec_now_us()) {
    for (auto& req : reqs) {
        req.*field = timestamp_us;
    }
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

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
static constexpr bool REMOTE_SPEC_NATIVE_LITTLE_ENDIAN = false;
#else
static constexpr bool REMOTE_SPEC_NATIVE_LITTLE_ENDIAN = true;
#endif

static constexpr uint32_t REMOTE_SPEC_REQUEST_FIELD_COUNT = 15;

template <typename T>
static inline void pack_optional_value(msgpack::packer<StringStream>& pk, const std::optional<T>& value) {
    if (value) pk.pack(*value);
    else pk.pack_nil();
}

static inline uint32_t remote_spec_load_le_u32(const char* src) {
    return static_cast<uint32_t>(static_cast<unsigned char>(src[0])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(src[1])) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(src[2])) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(src[3])) << 24);
}

static void pack_optional_int_vector_bin(
    msgpack::packer<StringStream>& pk,
    const std::optional<std::vector<int>>& values) {
    if (!values) {
        pk.pack_nil();
        return;
    }

    const auto& vec = *values;
    const uint32_t byte_size = static_cast<uint32_t>(vec.size() * sizeof(int32_t));
    pk.pack_bin(byte_size);

    if (vec.empty()) {
        pk.pack_bin_body("", 0);
        return;
    }

    if constexpr (sizeof(int) == sizeof(int32_t)) {
        if (REMOTE_SPEC_NATIVE_LITTLE_ENDIAN) {
            pk.pack_bin_body(reinterpret_cast<const char*>(vec.data()), byte_size);
            return;
        }
    }

    thread_local std::string scratch;
    scratch.resize(byte_size);
    char* dst = scratch.data();
    for (size_t i = 0; i < vec.size(); ++i) {
        int32_t value = static_cast<int32_t>(vec[i]);
        size_t offset = i * sizeof(int32_t);
        dst[offset + 0] = static_cast<char>(value & 0xff);
        dst[offset + 1] = static_cast<char>((value >> 8) & 0xff);
        dst[offset + 2] = static_cast<char>((value >> 16) & 0xff);
        dst[offset + 3] = static_cast<char>((value >> 24) & 0xff);
    }
    pk.pack_bin_body(scratch.data(), byte_size);
}

static void pack_optional_float_vector_bin(
    msgpack::packer<StringStream>& pk,
    const std::optional<std::vector<float>>& values) {
    if (!values) {
        pk.pack_nil();
        return;
    }

    const auto& vec = *values;
    const uint32_t byte_size = static_cast<uint32_t>(vec.size() * sizeof(float));
    pk.pack_bin(byte_size);

    if (vec.empty()) {
        pk.pack_bin_body("", 0);
        return;
    }

    if constexpr (sizeof(float) == 4) {
        if (REMOTE_SPEC_NATIVE_LITTLE_ENDIAN) {
            pk.pack_bin_body(reinterpret_cast<const char*>(vec.data()), byte_size);
            return;
        }
    }

    thread_local std::string scratch;
    scratch.resize(byte_size);
    char* dst = scratch.data();
    for (size_t i = 0; i < vec.size(); ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &vec[i], sizeof(bits));
        size_t offset = i * sizeof(float);
        dst[offset + 0] = static_cast<char>(bits & 0xff);
        dst[offset + 1] = static_cast<char>((bits >> 8) & 0xff);
        dst[offset + 2] = static_cast<char>((bits >> 16) & 0xff);
        dst[offset + 3] = static_cast<char>((bits >> 24) & 0xff);
    }
    pk.pack_bin_body(scratch.data(), byte_size);
}

static void pack_remote_spec_request(msgpack::packer<StringStream>& pk, const RemoteSpecRequest& req) {
    pk.pack_array(REMOTE_SPEC_REQUEST_FIELD_COUNT);
    pack_optional_value(pk, req.request_id);
    pack_optional_value(pk, req.spec_cnt);
    pk.pack(req.action);
    pk.pack(req.spec_type);
    pack_optional_int_vector_bin(pk, req.draft_token_ids);
    pack_optional_int_vector_bin(pk, req.input_ids);
    pack_optional_int_vector_bin(pk, req.output_ids);
    pack_optional_value(pk, req.num_draft_tokens);
    pack_optional_value(pk, req.sampling_params);
    pack_optional_value(pk, req.grammar);
    pk.pack(req.target_send_time);
    pk.pack(req.target_recv_time);
    pack_optional_float_vector_bin(pk, req.draft_logprobs);
    pk.pack(req.draft_recv_time);
    pk.pack(req.draft_send_time);
}

static std::string pack_remote_spec_batch_payload(const std::vector<RemoteSpecRequest>& objs) {
    thread_local size_t cached_capacity = 4096;
    std::string raw_data;
    raw_data.reserve(cached_capacity);
    StringStream ss(raw_data);
    msgpack::packer<StringStream> pk(ss);
    pk.pack_array(objs.size());
    for (const auto& obj : objs) {
        pack_remote_spec_request(pk, obj);
    }
    if (raw_data.capacity() > cached_capacity) cached_capacity = raw_data.capacity();
    return raw_data;
}

template <typename T>
static inline void unpack_optional_value(const msgpack::object& obj, std::optional<T>& value) {
    if (obj.is_nil()) {
        value.reset();
        return;
    }
    value = obj.as<T>();
}

static void unpack_optional_int_vector_bin(
    const msgpack::object& obj,
    std::optional<std::vector<int>>& values) {
    if (obj.is_nil()) {
        values.reset();
        return;
    }
    if (obj.type == msgpack::type::ARRAY) {
        values = obj.as<std::vector<int>>();
        return;
    }
    if (obj.type != msgpack::type::BIN) {
        throw std::runtime_error("invalid int vector payload type");
    }

    const auto& bin = obj.via.bin;
    if (bin.size % sizeof(int32_t) != 0) {
        throw std::runtime_error("invalid int vector payload size");
    }

    auto& vec = values.emplace();
    vec.resize(bin.size / sizeof(int32_t));
    if (vec.empty()) return;

    if constexpr (sizeof(int) == sizeof(int32_t)) {
        if (REMOTE_SPEC_NATIVE_LITTLE_ENDIAN) {
            std::memcpy(vec.data(), bin.ptr, bin.size);
            return;
        }
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<int>(static_cast<int32_t>(remote_spec_load_le_u32(bin.ptr + i * sizeof(int32_t))));
    }
}

static void unpack_optional_float_vector_bin(
    const msgpack::object& obj,
    std::optional<std::vector<float>>& values) {
    if (obj.is_nil()) {
        values.reset();
        return;
    }
    if (obj.type == msgpack::type::ARRAY) {
        values = obj.as<std::vector<float>>();
        return;
    }
    if (obj.type != msgpack::type::BIN) {
        throw std::runtime_error("invalid float vector payload type");
    }

    const auto& bin = obj.via.bin;
    if (bin.size % sizeof(float) != 0) {
        throw std::runtime_error("invalid float vector payload size");
    }

    auto& vec = values.emplace();
    vec.resize(bin.size / sizeof(float));
    if (vec.empty()) return;

    if constexpr (sizeof(float) == 4) {
        if (REMOTE_SPEC_NATIVE_LITTLE_ENDIAN) {
            std::memcpy(vec.data(), bin.ptr, bin.size);
            return;
        }
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        uint32_t bits = remote_spec_load_le_u32(bin.ptr + i * sizeof(float));
        std::memcpy(&vec[i], &bits, sizeof(bits));
    }
}

static void unpack_remote_spec_request(const msgpack::object& obj, RemoteSpecRequest& req) {
    if (obj.type != msgpack::type::ARRAY || obj.via.array.size < REMOTE_SPEC_REQUEST_FIELD_COUNT) {
        throw std::runtime_error("invalid remote spec request payload");
    }

    const msgpack::object* fields = obj.via.array.ptr;
    unpack_optional_value(fields[0], req.request_id);
    unpack_optional_value(fields[1], req.spec_cnt);
    req.action = fields[2].as<RemoteSpecAction>();
    req.spec_type = fields[3].as<SpecType>();
    unpack_optional_int_vector_bin(fields[4], req.draft_token_ids);
    unpack_optional_int_vector_bin(fields[5], req.input_ids);
    unpack_optional_int_vector_bin(fields[6], req.output_ids);
    unpack_optional_value(fields[7], req.num_draft_tokens);
    unpack_optional_value(fields[8], req.sampling_params);
    unpack_optional_value(fields[9], req.grammar);
    req.target_send_time = fields[10].as<double>();
    req.target_recv_time = fields[11].as<double>();
    unpack_optional_float_vector_bin(fields[12], req.draft_logprobs);
    req.draft_recv_time = fields[13].as<double>();
    req.draft_send_time = fields[14].as<double>();
}

static bool unpack_remote_spec_batch_payload(
    msgpack::unpacker& unpacker,
    const void* data,
    size_t len,
    std::vector<RemoteSpecRequest>& objs) {
    try {
        objs.clear();
        unpacker.reserve_buffer(len);
        std::memcpy(unpacker.buffer(), data, len);
        unpacker.buffer_consumed(len);

        msgpack::object_handle oh;
        if (!unpacker.next(oh)) {
            unpacker.remove_nonparsed_buffer();
            unpacker.reset();
            unpacker.reset_zone();
            return false;
        }

        const auto& root = oh.get();
        if (root.type != msgpack::type::ARRAY) {
            throw std::runtime_error("invalid remote spec batch payload");
        }

        objs.reserve(root.via.array.size);
        for (uint32_t i = 0; i < root.via.array.size; ++i) {
            objs.emplace_back();
            unpack_remote_spec_request(root.via.array.ptr[i], objs.back());
        }
        unpacker.remove_nonparsed_buffer();
        unpacker.reset_zone();
        return true;
    } catch (...) {
        unpacker.remove_nonparsed_buffer();
        unpacker.reset();
        unpacker.reset_zone();
        throw;
    }
}

static std::vector<RemoteSpecRequest> from_py_list(const py::list& objs) {
    std::vector<RemoteSpecRequest> out;
    out.reserve(static_cast<size_t>(py::len(objs)));
    for (auto item : objs) out.push_back(from_py_dict(item.cast<py::dict>()));
    return out;
}

template <typename T>
class BatchVectorPool {
public:
    using Batch = std::vector<T>;

    struct Reclaimer {
        BatchVectorPool* pool = nullptr;

        void operator()(Batch* batch) const {
            if (batch == nullptr) return;
            if (pool == nullptr) {
                delete batch;
                return;
            }
            pool->recycle(batch);
        }
    };

    using Handle = std::unique_ptr<Batch, Reclaimer>;

    explicit BatchVectorPool(size_t max_cached = 128) : max_cached_(max_cached) {}

    Handle acquire() {
        std::unique_ptr<Batch> batch;
        {
            std::lock_guard<std::mutex> lock(pool_mtx_);
            if (!free_list_.empty()) {
                batch = std::move(free_list_.back());
                free_list_.pop_back();
            }
        }
        if (!batch) batch = std::make_unique<Batch>();
        batch->clear();
        return Handle(batch.release(), Reclaimer{this});
    }

private:
    void recycle(Batch* batch) {
        batch->clear();
        std::unique_ptr<Batch> recycled(batch);
        std::lock_guard<std::mutex> lock(pool_mtx_);
        if (free_list_.size() >= max_cached_) return;
        free_list_.push_back(std::move(recycled));
    }

    size_t max_cached_;
    std::mutex pool_mtx_;
    std::vector<std::unique_ptr<Batch>> free_list_;
};

struct DealerRawBuffer {
    zmq::message_t payload;
};

struct RouterRawBuffer {
    std::string id;
    zmq::message_t payload;
};

static inline bool is_heartbeat_payload(const zmq::message_t& payload) {
    constexpr size_t heartbeat_size = sizeof("DEALER_HEARTBEAT") - 1;
    return payload.size() == heartbeat_size &&
           std::memcmp(payload.data(), "DEALER_HEARTBEAT", heartbeat_size) == 0;
}

// =====================================================
// 优化后的线程安全 SafeQueue (基于 RingBuffer 实现)
// =====================================================
template <typename T>
class SafeQueue {
public:
    // 使用 2 的幂次方大小以便进行位运算优化
    explicit SafeQueue(size_t capacity = 65536)
        : capacity_(capacity), mask_(capacity - 1), head_(0), tail_(0), size_(0), wake_epoch_(0) {
        // 确保 capacity 是 2 的幂
        if ((capacity & (capacity - 1)) != 0) {
            capacity_ = 65536;
            mask_ = capacity_ - 1;
        }
        buffer_.resize(capacity_);
    }
    void push(T&& item) {
        uint64_t wait_epoch = wake_epoch_.load(std::memory_order_acquire);
        std::unique_lock<std::mutex> lock(queue_mtx_);

        // 如果队列满了，这里采用阻塞/等待策略以维持 SafeQueue 的原有行为
        not_full_cv_.wait(lock, [this, wait_epoch] {
            return size_ < capacity_ || wake_epoch_.load(std::memory_order_acquire) != wait_epoch;
        });
        if (size_ == capacity_) {
            return;
        }

        buffer_[tail_] = std::move(item);
        tail_ = (tail_ + 1) & mask_;
        ++size_;
        lock.unlock();

        // 唤醒可能正在等待的 pop (兼容原有 CV 逻辑)
        not_empty_cv_.notify_one();
    }
    bool pop(T& item, int timeout_ms = 1) {
        uint64_t wait_epoch = wake_epoch_.load(std::memory_order_acquire);
        std::unique_lock<std::mutex> lock(queue_mtx_);

        if (size_ == 0) {
            // 队列为空，使用条件变量等待以节省 CPU
            if (!not_empty_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, wait_epoch] {
                return size_ > 0 || wake_epoch_.load(std::memory_order_acquire) != wait_epoch;
            })) {
                return false;
            }
            if (size_ == 0) {
                return false;
            }
        }

        item = std::move(buffer_[head_]);
        head_ = (head_ + 1) & mask_;
        --size_;
        lock.unlock();
        not_full_cv_.notify_one();
        return true;
    }
    // 辅助接口：唤醒所有阻塞在 pop 的线程，用于 stop()
    void notify_all() {
        wake_epoch_.fetch_add(1, std::memory_order_acq_rel);
        std::lock_guard<std::mutex> lock(queue_mtx_);
        not_empty_cv_.notify_all();
        not_full_cv_.notify_all();
    }
    // 辅助检查头部元素是否超时 (T 需包含 timestamp)
    bool peek_and_check_timeout(long long timeout_ms) {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        if (size_ == 0) return false;
        auto now = std::chrono::steady_clock::now();
        auto msg_time = buffer_[head_].timestamp;
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - msg_time).count();

        return diff > timeout_ms;
    }
    vector<T> pop_all() {
        vector<T> res;
        std::unique_lock<std::mutex> lock(queue_mtx_);
        res.reserve(size_);
        // 尽力而为地排空当前所有可用数据
        while (size_ > 0) {
            res.push_back(std::move(buffer_[head_]));
            head_ = (head_ + 1) & mask_;
            --size_;
        }
        lock.unlock();
        not_full_cv_.notify_all();
        return res;
    }
    size_t size() {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        return size_;
    }
    void clear() {
        std::unique_lock<std::mutex> lock(queue_mtx_);
        head_ = 0;
        tail_ = 0;
        size_ = 0;
        lock.unlock();
        not_full_cv_.notify_all();
    }
private:
    size_t capacity_;
    size_t mask_;
    std::vector<T> buffer_;

    // 使用 RingBuffer 结构配合互斥锁实现多生产者/多消费者安全
    size_t head_;
    size_t tail_;
    size_t size_;
    std::atomic<uint64_t> wake_epoch_;

    // 为了兼容原有代码的阻塞等待模式，保留辅助的 CV
    std::mutex queue_mtx_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
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
template <typename Derived, typename RawRecvT = string, typename RawSendT = string>
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
        if (bind_) {
            endpoint_type = "Router";
        } else {
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

        // 线程三：监控线程 处理 接收队列限流/ Draft 心跳状态监控
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
            send_raw_queue_.notify_all();
            internal_ctx_.shutdown();
            if (zmq_io_thread_.joinable()) zmq_io_thread_.join();
            if (zmq_data_unpack_thread_.joinable()) zmq_data_unpack_thread_.join();
            if (endpoint_monitor_thread_.joinable()) endpoint_monitor_thread_.join();
            sock_.close();
            internal_ctx_.close();
        } catch (...) {}
        cout << "[ZMQ " << endpoint_type << " Stop!] addr:" << addr_ << endl;
    }

    void enqueue_send_raw(RawSendT&& obj) {
        send_raw_queue_.push(move(obj));
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
        sock_.set(zmq::sockopt::immediate, 0);
        sock_.set(zmq::sockopt::sndbuf, 10 * 1024 * 1024);
        sock_.set(zmq::sockopt::rcvbuf, 10 * 1024 * 1024);
    }

    // 线程一：IO 循环
    void zmq_io_loop() {
        while (running_) {
            try {
                zmq_pollitem_t items[] = {{ static_cast<void*>(sock_), 0, ZMQ_POLLIN, 0 }};
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
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
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

    void endpoint_monitor_loop() {
        while (running_) {
            static_cast<Derived*>(this)->on_monitor_tick();
            std::this_thread::sleep_for(std::chrono::milliseconds(ENDPOINT_MONITOR_INTERNAL_MS));
        }
    }

    // 默认的 Multipart 接收逻辑
    void handle_raw_recv_base() {
        int max_batch = 100;
        bool should_stop_recv = false;
        while (max_batch-- > 0 && !should_stop_recv) {
            vector<zmq::message_t> frames;
            size_t total_bytes = 0;
            std::chrono::steady_clock::time_point recv_start = std::chrono::steady_clock::now();
            while (true) {
                zmq::message_t msg;
                auto res = this->sock_.recv(msg, zmq::recv_flags::dontwait);
                if (!res) {
                    should_stop_recv = true;
                    break;
                }
                total_bytes += msg.size();
                frames.push_back(std::move(msg));
                if (!this->sock_.get(zmq::sockopt::rcvmore)) break;
            }
            if (!frames.empty()) {
                std::chrono::steady_clock::time_point recv_end = std::chrono::steady_clock::now();
                remote_spec_debug_log(
                    "[ZMQ LOG C++][RECV] bytes=", total_bytes,
                    " time_us=", remote_spec_duration_us(recv_start, recv_end));
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
    thread endpoint_monitor_thread_;

    // 队列链路
    SafeQueue<RawRecvT> recv_raw_queue_;      // IO -> Codec
    SafeQueue<RawSendT> send_raw_queue_;      // Python/Monitor -> IO
};

// =====================================================
// DEALER Endpoint
// =====================================================
class [[gnu::visibility("default")]] DealerEndpoint : public AsyncZmqEndpointCRTP<DealerEndpoint, DealerRawBuffer, string> {
public:
    using RequestBatchHandle = typename BatchVectorPool<RemoteSpecRequest>::Handle;

    DealerEndpoint(const string& addr, const string& identity, bool bind = false)
        : AsyncZmqEndpointCRTP(addr, bind, zmq::socket_type::dealer), identity_(identity) {}

    void do_setup_socket() {
        this->sock_.set(zmq::sockopt::routing_id, identity_);
        this->setup_common_socket_options();
    }

    void on_reconnect() { send_heartbeat(); }

    void send_heartbeat() {
        this->enqueue_send_raw(string(DEALER_HEARTBEAT));
    }

    void send_objs(const vector<RemoteSpecRequest>& reqs) {
        auto copied_reqs = reqs;
        send_objs(std::move(copied_reqs));
    }

    void send_objs(vector<RemoteSpecRequest>&& reqs) {
        stamp_request_batch(reqs, &RemoteSpecRequest::draft_send_time);
        // pack + send
        std::chrono::steady_clock::time_point pack_start = std::chrono::steady_clock::now();
        auto packed = pack_remote_spec_batch_payload(reqs);
        std::chrono::steady_clock::time_point pack_end = std::chrono::steady_clock::now();
        remote_spec_debug_log(
            "[ZMQ LOG C++][PACK] nums=", reqs.size(),
            " bytes=", packed.size(),
            " time_us=", remote_spec_duration_us(pack_start, pack_end));
        this->enqueue_send_raw(std::move(packed));
    }

    vector<RemoteSpecRequest> get_received_objs() {
        auto wrapped_list = python_ready_queue_.pop_all();
        size_t total_size = 0;
        for (auto& wrapped : wrapped_list) total_size += wrapped.data->size();

        vector<RemoteSpecRequest> out;
        out.reserve(total_size);
        for (auto& wrapped : wrapped_list) {
            for (auto& obj : *wrapped.data) out.push_back(std::move(obj));
        }
        return out;
    }

    void on_monitor_tick() {
        if (python_ready_queue_.peek_and_check_timeout(PYTHON_READY_QUEUE_TIMEOUT_MS)) {
            while (python_ready_queue_.peek_and_check_timeout(PYTHON_READY_QUEUE_TIMEOUT_MS)) {
                TimestampedObj<RequestBatchHandle> discarded;
                if (python_ready_queue_.pop(discarded, 0)) {
                    if ( remote_spec_debug_enabled() ) {
                        cout << "[ZMQ LOG C++][Monitor] " << endpoint_type << " 接收打包数据超时未被处理，已丢弃！" << endl;
                    }
                }
            }
        }
        send_heartbeat();
        if ( remote_spec_debug_enabled() ) {
            cout << "[ZMQ LOG C++] " << identity_ << " 发送心跳信号到 " << addr_ << endl;
        }
    }

    void handle_raw_recv() { this->handle_raw_recv_base(); }

    void dispatch_frames(vector<zmq::message_t>&& frames) {
        // Dealer 通常只关心最后一帧作为 payload
        DealerRawBuffer raw;
        raw.payload = std::move(frames.back());
        recv_raw_queue_.push(std::move(raw));
    }

    void handle_raw_send(string& data) {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        this->sock_.send(zmq::buffer(data), zmq::send_flags::none);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        remote_spec_debug_log(
            "[ZMQ LOG C++][SEND] bytes=", data.size(),
            " time_us=", remote_spec_duration_us(start, end));
    }

    void process_incoming_data(DealerRawBuffer&& raw) {
        try {
            auto reqs = request_batch_pool_.acquire();
            std::chrono::steady_clock::time_point unpack_start = std::chrono::steady_clock::now();
            bool ok = unpack_remote_spec_batch_payload(batch_unpacker_, raw.payload.data(), raw.payload.size(), *reqs);
            std::chrono::steady_clock::time_point unpack_end = std::chrono::steady_clock::now();
            if (!ok) {
                if ( remote_spec_debug_enabled() ) cerr << "[Dealer Unpack Error]: incomplete batch payload" << endl;
                return;
            }
            remote_spec_debug_log(
                "[ZMQ LOG C++][UNPACK] nums=", reqs->size(),
                " bytes=", raw.payload.size(),
                " time_us=", remote_spec_duration_us(unpack_start, unpack_end));
            stamp_request_batch(*reqs, &RemoteSpecRequest::draft_recv_time);
            // recv + unpack
            python_ready_queue_.push({std::move(reqs), chrono::steady_clock::now()});
        } catch (const std::exception& e) {
            if ( remote_spec_debug_enabled() ) cerr << "[Dealer Unpack Error]: " << e.what() << endl;
        }
    }

private:
    string identity_;
    msgpack::unpacker batch_unpacker_;
    BatchVectorPool<RemoteSpecRequest> request_batch_pool_;
    SafeQueue<TimestampedObj<RequestBatchHandle>> python_ready_queue_;
};

// =====================================================
// ROUTER Endpoint
// =====================================================
class [[gnu::visibility("default")]] RouterEndpoint : public AsyncZmqEndpointCRTP<RouterEndpoint, RouterRawBuffer, pair<string, string>> {
public:
    using RequestBatchHandle = typename BatchVectorPool<RemoteSpecRequest>::Handle;

    RouterEndpoint(const string& addr, bool bind = true)
        : AsyncZmqEndpointCRTP(addr, bind, zmq::socket_type::router) {}

    void do_setup_socket() { this->setup_common_socket_options(); }
    void on_reconnect() {}

    void send_objs(const string& id, const vector<RemoteSpecRequest>& reqs) {
        auto copied_reqs = reqs;
        send_objs(id, std::move(copied_reqs));
    }

    void send_objs(const string& id, vector<RemoteSpecRequest>&& reqs) {
        stamp_request_batch(reqs, &RemoteSpecRequest::target_send_time);
        std::chrono::steady_clock::time_point pack_start = std::chrono::steady_clock::now();
        auto packed = pack_remote_spec_batch_payload(reqs);
        std::chrono::steady_clock::time_point pack_end = std::chrono::steady_clock::now();
        remote_spec_debug_log(
            "[ZMQ LOG C++][PACK] nums=", reqs.size(),
            " bytes=", packed.size(),
            " time_us=", remote_spec_duration_us(pack_start, pack_end));
        this->enqueue_send_raw({id, std::move(packed)});
    }

    vector<pair<string, RemoteSpecRequest>> get_received_objs() {
        auto wrapped_list = python_ready_queue_.pop_all();
        size_t total_size = 0;
        for (auto& wrapped : wrapped_list) total_size += wrapped.data.second->size();

        vector<pair<string, RemoteSpecRequest>> out;
        out.reserve(total_size);
        for (auto& wrapped : wrapped_list) {
            const string& id = wrapped.data.first;
            for (auto& obj : *wrapped.data.second) out.push_back({id, std::move(obj)});
        }
        return out;
    }

    void on_monitor_tick() {
        if (python_ready_queue_.peek_and_check_timeout(PYTHON_READY_QUEUE_TIMEOUT_MS)) {
            while (python_ready_queue_.peek_and_check_timeout(PYTHON_READY_QUEUE_TIMEOUT_MS)) {
                TimestampedObj<pair<string, RequestBatchHandle>> discarded;
                if (python_ready_queue_.pop(discarded, 0)) {
                    if ( remote_spec_debug_enabled() ) cout << "[ZMQ LOG C++][Monitor] " << endpoint_type << " 接收打包数据超时未被处理，已丢弃！" << endl;
                }
            }
        }
        auto now = chrono::steady_clock::now();
        lock_guard<mutex> lk(reg_mtx_);
        auto it = registered_dealers_.begin();
        while (it != registered_dealers_.end()) {
            if (chrono::duration_cast<chrono::milliseconds>(now - it->second).count() > DEALER_HEARTBEAT_TIMEOUT_MS) {
                if ( remote_spec_debug_enabled() ) cout << "[ZMQ LOG C++] " << it->first << " 心跳信号超时，剔除!" << endl;
                it = registered_dealers_.erase(it);
            } else ++it;
        }
    }

    void handle_raw_recv() { this->handle_raw_recv_base(); }

    void dispatch_frames(vector<zmq::message_t>&& frames) {
        if (frames.size() < 2) return;
        string id(static_cast<char*>(frames[0].data()), frames[0].size());
        auto& payload = frames.back();

        if (is_heartbeat_payload(payload)) {
            lock_guard<mutex> lk(reg_mtx_);
            registered_dealers_[id] = chrono::steady_clock::now();
            if ( remote_spec_debug_enabled() ) cout << "[ZMQ LOG C++]" << addr_ << " 收到心跳信号，来自 " << id << endl;
        } else {
            RouterRawBuffer raw;
            raw.id = std::move(id);
            raw.payload = std::move(payload);
            recv_raw_queue_.push(std::move(raw));
        }
    }

    void handle_raw_send(pair<string, string>& data) {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        this->sock_.send(zmq::buffer(data.first), zmq::send_flags::sndmore);
        this->sock_.send(zmq::message_t{}, zmq::send_flags::sndmore);
        this->sock_.send(zmq::buffer(data.second), zmq::send_flags::none);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        remote_spec_debug_log(
            "[ZMQ LOG C++][SEND] bytes=", data.second.size(),
            " time_us=", remote_spec_duration_us(start, end));
    }

    void process_incoming_data(RouterRawBuffer&& raw) {
        try {
            auto reqs = request_batch_pool_.acquire();
            std::chrono::steady_clock::time_point unpack_start = std::chrono::steady_clock::now();
            bool ok = unpack_remote_spec_batch_payload(batch_unpacker_, raw.payload.data(), raw.payload.size(), *reqs);
            std::chrono::steady_clock::time_point unpack_end = std::chrono::steady_clock::now();
            if (!ok) {
                if ( remote_spec_debug_enabled() ) cerr << "[Router Unpack Error]: incomplete batch payload" << endl;
                return;
            }
            remote_spec_debug_log(
                "[ZMQ LOG C++][UNPACK] nums=", reqs->size(),
                " bytes=", raw.payload.size(),
                " time_us=", remote_spec_duration_us(unpack_start, unpack_end));
            stamp_request_batch(*reqs, &RemoteSpecRequest::target_recv_time);
            python_ready_queue_.push({{std::move(raw.id), std::move(reqs)}, chrono::steady_clock::now()});
        } catch (const std::exception& e) {
            if ( remote_spec_debug_enabled() ) cerr << "[Router Unpack Error]: " << e.what() << endl;
        }
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
    msgpack::unpacker batch_unpacker_;
    BatchVectorPool<RemoteSpecRequest> request_batch_pool_;
    SafeQueue<TimestampedObj<pair<string, RequestBatchHandle>>> python_ready_queue_;
};

// =====================================================
// PYBIND11
// =====================================================
PYBIND11_MODULE(remote_spec_zmq, m) {
    py::class_<DealerEndpoint>(m, "DealerEndpoint")
        .def(py::init<const string&, const string&, bool>())
        .def("start", &DealerEndpoint::start)
        .def("stop", &DealerEndpoint::stop)
        .def("send_objs", [](DealerEndpoint& self, py::list objs) {
            auto reqs = from_py_list(objs);
            {
                py::gil_scoped_release release;
                self.send_objs(std::move(reqs));
            }
        })
        .def("get_received_objs", [](DealerEndpoint& self) {
            py::list out;
            auto datas = [&self]() {
                py::gil_scoped_release release;
                return self.get_received_objs();
            }();
            for (auto& obj : datas) out.append(to_py_dict(obj));
            return out;
        });

    py::class_<RouterEndpoint>(m, "RouterEndpoint")
        .def(py::init<const string&, bool>())
        .def("start", &RouterEndpoint::start)
        .def("stop", &RouterEndpoint::stop)
        .def("get_all_dealers", &RouterEndpoint::get_all_dealers)
        .def("send_objs", [](RouterEndpoint& self, const string& id, py::list objs) {
            auto reqs = from_py_list(objs);
            {
                py::gil_scoped_release release;
                self.send_objs(id, std::move(reqs));
            }
        })
        .def("get_received_objs", [](RouterEndpoint& self) {
            py::list out;
            auto datas = [&self]() {
                py::gil_scoped_release release;
                return self.get_received_objs();
            }();
            for (auto& p : datas) out.append(py::make_tuple(p.first, to_py_dict(p.second)));
            return out;
        });
}
