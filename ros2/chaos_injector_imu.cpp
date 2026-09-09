// chaos_injector_imu.cpp
//
// Injects delay, jitter, bursts, drops, duplicates, reordering and clock
// skew into an IMU stream so downstream nodes can be tested against a
// hostile sensor feed.
//
// ROS2 is not available in the environment where this revision was written,
// so the node itself was not built. The scheduling logic was extracted into
// test_drain_order.cpp (plain C++, no rclcpp) and compiled and run; the
// numbers quoted below come from that harness.
//
// ---------------------------------------------------------------------
// FIX #1 (primary): drain() published in buffer order, not in due order,
// which silently defeated the reorder feature it was supposed to drive.
//
//   for (auto it = buf_.begin(); it != buf_.end();) {
//       if (it->due <= now) { publish(*it->msg); it = buf_.erase(it); }
//       else ++it;
//   }
//
// Every message whose due time has passed is published during the same
// timer tick, in the order it happens to sit in the deque -- i.e. arrival
// order. Two messages whose due times differ by less than one tick period
// are therefore always delivered in arrival order, whatever their due
// times say. Measured over 20000 messages at 200 Hz with a 5 ms timer and
// the file's own delay parameters (-10..+20 ms uniform, 2 ms jitter):
//   - 1459 delivered pairs were in the wrong order with respect to their
//     own due times;
//   - of the reorder swaps performed by the previous revision's fix,
//     30.1% had no observable effect at all, because both messages landed
//     in the same tick and the deque order decided the outcome.
// The previous revision changed std::swap(buf_[i], buf_[j]) into a swap of
// due times only, which was the right call, but the swap can only matter
// if delivery order actually follows due time. It did not.
// Fixed by collecting the ready messages, sorting them by due time
// (stable, so equal due times keep arrival order) and publishing in that
// order. The harness reports 0 due-order violations afterwards.
//
// FIX #2 (memory safety): the reorder index could run off the end.
//   size_t i = std::floor(rng_->uniform(0, (double)buf_.size()));
// If ChaosRNG::uniform is inclusive of its upper bound, or returns exactly
// the bound through rounding, this yields buf_[size()], which is undefined
// behaviour on a deque. The draw is now clamped to size()-1, and the i==j
// case (a swap that does nothing) is rejected and retried once.
//
// FIX #3 (unbounded growth): buf_ had no capacity limit. If the timer is
// starved, or hold_publish is on while burst_delay_ms is large, the buffer
// grows without bound and the node's memory grows with it -- in a chaos
// test that failure looks like a downstream fault. A max_buffer parameter
// now drops the oldest entry and counts it.
//
// FIX #4 (QoS mismatch): the subscription used SensorDataQoS (best effort,
// small depth) while the publisher used the default reliable QoS with
// depth 50. A chaos injector should not quietly upgrade the reliability of
// the stream under test, since retransmission changes the very timing the
// test is about. The publisher now uses SensorDataQoS by default, with a
// reliable_output parameter for the cases where that is wanted.
//
// FIX #5 (thread safety): buf_ and rng_ are touched from both the
// subscription callback and the timer callback. That is safe under the
// default single-threaded executor and a data race under a multi-threaded
// one. A mutex now guards both, so the node cannot become a source of
// nondeterminism in tests that are hunting nondeterminism.
//
// FIX #6 (invalid input stamps): some drivers publish a zero header.stamp.
// Adding an offset to zero produced a timestamp near the epoch, which
// downstream filters typically reject outright, making the injector look
// like a broken link rather than a delayed one. A zero stamp is now
// replaced with the current time before the offset is applied.
//
// FIX #7 (observability): the node reported nothing about what it did, so
// a downstream failure could not be attributed to a specific injected
// fault. Counters for received / dropped / duplicated / reordered /
// buffer-overflow messages are now logged periodically.
//
// FIX #8 (magic numbers, runtime tuning): the 0.3 reorder probability and
// the 5 ms timer period were hard-coded. Both are parameters now, and a
// parameter callback allows tuning during a run, which is the normal way
// this kind of node is used.
//
// Unchanged: the elapsed-time-based skew model and its stamp-only
// application, and the drop/dup/burst semantics.
// ---------------------------------------------------------------------

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "psf_zero_eit/chaos_utils.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <mutex>
#include <vector>

using sensor_msgs::msg::Imu;

struct ImuItem {
  Imu::SharedPtr msg;
  rclcpp::Time due;
  rclcpp::Time orig_stamp;
  uint64_t seq;        // arrival order, used as a stable tie-break
};

class ChaosInjectorImu : public rclcpp::Node {
public:
  ChaosInjectorImu() : Node("chaos_injector_imu") {
    in_topic_  = declare_parameter<std::string>("in_topic", "/imu/raw");
    out_topic_ = declare_parameter<std::string>("out_topic", "/chaos/imu/raw");

    // IMU runs fast, so the delay window is kept narrow.
    prm_.delay_min_ms      = declare_parameter<double>("delay_min_ms", -10.0);
    prm_.delay_max_ms      = declare_parameter<double>("delay_max_ms", +20.0);
    prm_.jitter_sigma_ms   = declare_parameter<double>("jitter_sigma_ms", 2.0);
    prm_.burst_prob        = declare_parameter<double>("burst_prob", 0.01);
    prm_.burst_delay_ms    = declare_parameter<double>("burst_delay_ms", 50.0);
    prm_.drop_prob         = declare_parameter<double>("drop_prob", 0.005);
    prm_.dup_prob          = declare_parameter<double>("dup_prob", 0.001);
    prm_.reorder_window_ms = declare_parameter<double>("reorder_window_ms", 10.0);
    // Simulates the temperature drift of a low-cost MEMS part.
    prm_.skew_ppm          = declare_parameter<double>("skew_ppm", 100.0);
    prm_.hold_publish      = declare_parameter<bool>("hold_publish", true);
    prm_.stamp_only        = declare_parameter<bool>("stamp_only", false);
    prm_.seed              = static_cast<uint32_t>(declare_parameter<int>("seed", 0));

    reorder_prob_    = declare_parameter<double>("reorder_prob", 0.3);
    dup_delay_ms_    = declare_parameter<double>("dup_delay_ms", 1.0);
    max_buffer_      = static_cast<size_t>(declare_parameter<int>("max_buffer", 2000));
    tick_ms_         = declare_parameter<double>("tick_ms", 5.0);
    stats_period_s_  = declare_parameter<double>("stats_period_s", 10.0);
    reliable_output_ = declare_parameter<bool>("reliable_output", false);

    validate_parameters();

    rng_ = std::make_unique<psf::ChaosRNG>(prm_.seed);
    // Reference point for elapsed-time-based skew.
    start_time_ = this->get_clock()->now();

    // Match the input QoS by default: a chaos injector must not silently
    // make the stream under test more reliable than it really is.
    auto qos = reliable_output_ ? rclcpp::QoS(50).reliable() : rclcpp::QoS(rclcpp::SensorDataQoS());

    sub_ = create_subscription<Imu>(
        in_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ChaosInjectorImu::cb, this, std::placeholders::_1));
    pub_ = create_publisher<Imu>(out_topic_, qos);

    timer_ = create_wall_timer(
        std::chrono::microseconds(static_cast<int64_t>(tick_ms_ * 1000.0)),
        std::bind(&ChaosInjectorImu::drain, this));
    stats_timer_ = create_wall_timer(
        std::chrono::milliseconds(static_cast<int64_t>(stats_period_s_ * 1000.0)),
        std::bind(&ChaosInjectorImu::log_stats, this));

    param_cb_ = add_on_set_parameters_callback(
        std::bind(&ChaosInjectorImu::on_set_parameters, this, std::placeholders::_1));

    RCLCPP_WARN(get_logger(), "IMU chaos injector: %s -> %s (seed=%u)",
                in_topic_.c_str(), out_topic_.c_str(), prm_.seed);
  }

private:
  // ---------------- parameters and state ----------------
  psf::ChaosParams prm_;
  std::unique_ptr<psf::ChaosRNG> rng_;
  std::string in_topic_, out_topic_;
  double reorder_prob_{0.3};
  double dup_delay_ms_{1.0};
  size_t max_buffer_{2000};
  double tick_ms_{5.0};
  double stats_period_s_{10.0};
  bool reliable_output_{false};

  rclcpp::Subscription<Imu>::SharedPtr sub_;
  rclcpp::Publisher<Imu>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_, stats_timer_;
  OnSetParametersCallbackHandle::SharedPtr param_cb_;

  std::mutex mtx_;                 // guards buf_, rng_ and the counters
  std::deque<ImuItem> buf_;
  rclcpp::Time start_time_;        // per-instance reference for skew
  uint64_t seq_{0};

  struct Stats {
    uint64_t received{0}, dropped{0}, duplicated{0}, published{0};
    uint64_t reordered{0}, overflow{0}, bursts{0};
    size_t buffer_high_water{0};
  } stats_;

  void validate_parameters() {
    if (prm_.delay_max_ms < prm_.delay_min_ms)
      throw std::invalid_argument("delay_max_ms must not be below delay_min_ms");
    if (tick_ms_ <= 0.0)
      throw std::invalid_argument("tick_ms must be positive");
    if (max_buffer_ < 2)
      throw std::invalid_argument("max_buffer must be at least 2");
  }

  rcl_interfaces::msg::SetParametersResult
  on_set_parameters(const std::vector<rclcpp::Parameter> &params) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto &p : params) {
      const auto &n = p.get_name();
      if      (n == "drop_prob")         prm_.drop_prob = p.as_double();
      else if (n == "dup_prob")          prm_.dup_prob = p.as_double();
      else if (n == "burst_prob")        prm_.burst_prob = p.as_double();
      else if (n == "burst_delay_ms")    prm_.burst_delay_ms = p.as_double();
      else if (n == "delay_min_ms")      prm_.delay_min_ms = p.as_double();
      else if (n == "delay_max_ms")      prm_.delay_max_ms = p.as_double();
      else if (n == "jitter_sigma_ms")   prm_.jitter_sigma_ms = p.as_double();
      else if (n == "skew_ppm")          prm_.skew_ppm = p.as_double();
      else if (n == "reorder_prob")      reorder_prob_ = p.as_double();
      else if (n == "reorder_window_ms") prm_.reorder_window_ms = p.as_double();
    }
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = (prm_.delay_max_ms >= prm_.delay_min_ms);
    if (!result.successful) result.reason = "delay_max_ms must not be below delay_min_ms";
    return result;
  }

  // ---------------- subscription ----------------
  void cb(const Imu::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    ++stats_.received;

    if (rng_->bernoulli(prm_.drop_prob)) { ++stats_.dropped; return; }

    double d_ms = rng_->uniform(prm_.delay_min_ms, prm_.delay_max_ms)
                + rng_->normal(0.0, prm_.jitter_sigma_ms);
    if (rng_->bernoulli(prm_.burst_prob)) { d_ms += prm_.burst_delay_ms; ++stats_.bursts; }

    // Skew follows elapsed wall-clock time, so it does not depend on the
    // topic's publish rate, and it is applied to the reported timestamp
    // only -- a drifting clock never makes a packet physically arrive late.
    const rclcpp::Time now = this->get_clock()->now();
    const double elapsed_ms = (now - start_time_).nanoseconds() / 1e6;
    const double skew_ms = (prm_.skew_ppm * 1e-6) * elapsed_ms;

    auto out = std::make_shared<Imu>(*msg);

    // A zero stamp from the driver would otherwise become a timestamp near
    // the epoch, which downstream filters reject as garbage rather than
    // treating as delayed.
    rclcpp::Time orig_stamp(msg->header.stamp, RCL_ROS_TIME);
    if (orig_stamp.nanoseconds() == 0) {
      orig_stamp = now;
      RCLCPP_WARN_ONCE(get_logger(), "input header.stamp is zero; substituting node time");
    }

    const double stamped_total_ms = d_ms + skew_ms;
    out->header.stamp = rclcpp::Time(
        orig_stamp.nanoseconds() + static_cast<int64_t>(stamped_total_ms * 1e6),
        RCL_ROS_TIME);

    if (prm_.stamp_only || !prm_.hold_publish) {
      pub_->publish(*out);
      ++stats_.published;
      if (rng_->bernoulli(prm_.dup_prob)) { pub_->publish(*out); ++stats_.duplicated; }
      return;
    }

    const double hold_ms = std::max(0.0, d_ms);
    const rclcpp::Time due = now + rclcpp::Duration::from_nanoseconds(
        static_cast<int64_t>(hold_ms * 1e6));

    if (buf_.size() >= max_buffer_) {
      buf_.pop_front();
      ++stats_.overflow;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "delay buffer full (%zu); dropping oldest", max_buffer_);
    }
    buf_.push_back({out, due, orig_stamp, seq_++});
    stats_.buffer_high_water = std::max(stats_.buffer_high_water, buf_.size());
  }

  // ---------------- timer ----------------
  void drain() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (buf_.empty()) return;

    const rclcpp::Time now = this->get_clock()->now();

    // Collect everything that is due, then publish in due order. Publishing
    // straight out of the deque delivers same-tick messages in arrival
    // order, which cancels both the randomized delays and the reorder
    // feature below.
    std::vector<ImuItem> ready;
    for (auto it = buf_.begin(); it != buf_.end();) {
      if (it->due <= now) { ready.push_back(*it); it = buf_.erase(it); }
      else ++it;
    }

    std::stable_sort(ready.begin(), ready.end(),
                     [](const ImuItem &a, const ImuItem &b) {
                       if (a.due == b.due) return a.seq < b.seq;  // stable tie-break
                       return a.due < b.due;
                     });

    for (const auto &item : ready) {
      pub_->publish(*(item.msg));
      ++stats_.published;
      if (rng_->bernoulli(prm_.dup_prob)) {
        // Real duplicates arrive slightly apart rather than back to back.
        auto dup = std::make_shared<Imu>(*(item.msg));
        dup->header.stamp = rclcpp::Time(
            rclcpp::Time(dup->header.stamp, RCL_ROS_TIME).nanoseconds()
              + static_cast<int64_t>(dup_delay_ms_ * 1e6),
            RCL_ROS_TIME);
        pub_->publish(*dup);
        ++stats_.duplicated;
      }
    }

    maybe_reorder();
  }

  // Trade delivery times between two buffered messages.
  void maybe_reorder() {
    if (buf_.size() < 2) return;
    const double span_ms = (buf_.back().due - buf_.front().due).seconds() * 1000.0;
    if (span_ms <= 0.0 || span_ms > prm_.reorder_window_ms) return;
    if (!rng_->bernoulli(reorder_prob_)) return;

    const size_t n = buf_.size();
    size_t i = draw_index(n);
    size_t j = draw_index(n);
    if (i == j) j = draw_index(n);   // one retry; a self-swap does nothing
    if (i == j) return;

    // Swap due times only. Swapping whole entries leaves delivery unchanged,
    // because ordering is decided by each item's own due time.
    std::swap(buf_[i].due, buf_[j].due);
    ++stats_.reordered;
  }

  // Clamped so a uniform draw that touches its upper bound cannot index
  // one past the end.
  size_t draw_index(size_t n) const {
    const double u = rng_->uniform(0.0, static_cast<double>(n));
    const auto idx = static_cast<size_t>(std::floor(u));
    return std::min(idx, n - 1);
  }

  void log_stats() {
    std::lock_guard<std::mutex> lock(mtx_);
    RCLCPP_INFO(get_logger(),
                "rx=%lu pub=%lu drop=%lu dup=%lu burst=%lu reorder=%lu "
                "overflow=%lu buf=%zu (peak %zu)",
                stats_.received, stats_.published, stats_.dropped,
                stats_.duplicated, stats_.bursts, stats_.reordered,
                stats_.overflow, buf_.size(), stats_.buffer_high_water);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ChaosInjectorImu>());
  rclcpp::shutdown();
  return 0;
}
