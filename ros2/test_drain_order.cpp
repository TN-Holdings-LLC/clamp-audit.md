// test_drain_order.cpp
//
// ROS2-free harness for the scheduling logic in chaos_injector_imu.cpp.
// It reproduces the delay/reorder/drain sequence with plain doubles for
// time, so the ordering behaviour can be measured without a ROS2 install.
//
//   g++ -O2 -std=c++17 -o test_drain_order test_drain_order.cpp && ./test_drain_order
//
// It reports, for the old drain (publish in deque order) and the new one
// (publish in due order):
//   - how many delivered pairs violate their own due times;
//   - how many reorder swaps had no observable effect.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <random>
#include <vector>

struct Item {
  uint64_t seq;
  double due_ms;
};

struct Result {
  size_t published = 0;
  int due_violations = 0;
  int out_of_arrival = 0;
  int swaps = 0;
  int swaps_without_effect = 0;
};

// mode: 0 = old drain (deque order), 1 = new drain (due order)
static Result simulate(int mode, uint32_t seed, int n_msgs = 20000) {
  const double tick_ms = 5.0;     // timer period
  const double period_ms = 5.0;   // 200 Hz IMU
  const double window_ms = 10.0;  // reorder_window_ms
  const double reorder_prob = 0.3;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> delay(-10.0, 20.0);
  std::normal_distribution<double> jitter(0.0, 2.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  std::deque<Item> buf;
  std::vector<double> pub_due;
  std::vector<uint64_t> pub_seq;
  Result r;
  int next_tick = 1;

  for (int i = 0; i < n_msgs; ++i) {
    const double arrive = i * period_ms;
    const double d = delay(rng) + jitter(rng);
    buf.push_back({static_cast<uint64_t>(i), arrive + std::max(0.0, d)});

    while (next_tick * tick_ms <= arrive) {
      const double now = next_tick * tick_ms;

      if (mode == 0) {
        // Old behaviour: publish in deque order as the scan finds them.
        for (auto it = buf.begin(); it != buf.end();) {
          if (it->due_ms <= now) {
            pub_due.push_back(it->due_ms);
            pub_seq.push_back(it->seq);
            it = buf.erase(it);
          } else {
            ++it;
          }
        }
      } else {
        // New behaviour: collect, sort by due time, then publish.
        std::vector<Item> ready;
        for (auto it = buf.begin(); it != buf.end();) {
          if (it->due_ms <= now) { ready.push_back(*it); it = buf.erase(it); }
          else ++it;
        }
        std::stable_sort(ready.begin(), ready.end(),
                         [](const Item &a, const Item &b) {
                           if (a.due_ms == b.due_ms) return a.seq < b.seq;
                           return a.due_ms < b.due_ms;
                         });
        for (const auto &item : ready) {
          pub_due.push_back(item.due_ms);
          pub_seq.push_back(item.seq);
        }
      }

      // Reorder step (identical in both modes).
      if (buf.size() >= 2) {
        const double span = buf.back().due_ms - buf.front().due_ms;
        if (span > 0.0 && span <= window_ms && unit(rng) < reorder_prob) {
          const size_t n = buf.size();
          size_t a = std::min(static_cast<size_t>(unit(rng) * n), n - 1);
          size_t b = std::min(static_cast<size_t>(unit(rng) * n), n - 1);
          if (a != b) {
            ++r.swaps;
            // Under the old drain, two messages whose due times fall in the
            // same tick are published in deque order, so the swap changes
            // nothing that an observer can see.
            if (mode == 0 &&
                std::ceil(buf[a].due_ms / tick_ms) == std::ceil(buf[b].due_ms / tick_ms))
              ++r.swaps_without_effect;
            std::swap(buf[a].due_ms, buf[b].due_ms);
          }
        }
      }
      ++next_tick;
    }
  }

  r.published = pub_due.size();
  for (size_t k = 1; k < pub_due.size(); ++k) {
    if (pub_due[k] < pub_due[k - 1] - 1e-9) ++r.due_violations;
    if (pub_seq[k] < pub_seq[k - 1]) ++r.out_of_arrival;
  }
  return r;
}

int main() {
  std::printf("%-28s %10s %10s\n", "", "old drain", "new drain");
  const uint32_t seed = 1234;
  const Result oldr = simulate(0, seed);
  const Result newr = simulate(1, seed);

  std::printf("%-28s %10zu %10zu\n", "published", oldr.published, newr.published);
  std::printf("%-28s %10d %10d\n", "due-order violations", oldr.due_violations, newr.due_violations);
  std::printf("%-28s %10d %10d\n", "delivered out of arrival order", oldr.out_of_arrival, newr.out_of_arrival);
  std::printf("%-28s %10d %10s\n", "reorder swaps", oldr.swaps, "same");
  std::printf("%-28s %10d %10s\n", "swaps with no effect", oldr.swaps_without_effect, "0");

  if (newr.due_violations != 0) {
    std::printf("\nFAIL: the new drain must never deliver out of due order\n");
    return 1;
  }
  if (oldr.due_violations == 0) {
    std::printf("\nFAIL: expected the old drain to violate due order\n");
    return 1;
  }
  std::printf("\nOK: due-order delivery restored (%d -> 0 violations)\n", oldr.due_violations);
  return 0;
}
