// A small persistent thread pool that splits one large host memcpy across a
// few threads (snapshot staging copies, state_io.cpp). A single core cannot
// saturate the memory system and its copy speed follows the core clock; a
// few cores do better. The result is byte-identical to std::memcpy.
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace gufo::models::ling_3_flash::rocm {

class HostCopyPool {
public:
  /// Copies below this size are not worth waking the workers for.
  static constexpr std::size_t kMinParallelBytes = std::size_t{2} << 20;
  /// Slices start on page boundaries relative to the first byte.
  static constexpr std::size_t kSliceAlign = 4096;

  /// `threads` counts the calling thread; `threads - 1` workers are started.
  /// If a thread cannot be started the pool simply has fewer.
  explicit HostCopyPool(unsigned threads) {
    try {
      if (threads > 1) {
        workers_.reserve(threads - 1);
      }
      for (unsigned i = 1; i < threads; ++i) {
        workers_.emplace_back([this, i] { Work(i); });
      }
    } catch (...) {
      // Keep the workers that started.
    }
  }

  ~HostCopyPool() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    work_ready_.notify_all();
    for (std::thread& t : workers_) {
      t.join();
    }
  }

  HostCopyPool(const HostCopyPool&) = delete;
  HostCopyPool& operator=(const HostCopyPool&) = delete;

  /// Same contract as std::memcpy (no overlap). Concurrent callers are
  /// serialized. Does not throw.
  void Copy(void* dst, const void* src, std::size_t bytes) noexcept {
    if (workers_.empty() || bytes < kMinParallelBytes) {
      std::memcpy(dst, src, bytes);
      return;
    }
    const std::lock_guard<std::mutex> caller(caller_mutex_);
    const std::size_t parts = workers_.size() + 1;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      dst_ = static_cast<std::uint8_t*>(dst);
      src_ = static_cast<const std::uint8_t*>(src);
      bytes_ = bytes;
      active_ = parts;
      pending_ = workers_.size();
      ++generation_;
    }
    work_ready_.notify_all();
    CopySlice(0, parts, static_cast<std::uint8_t*>(dst),
              static_cast<const std::uint8_t*>(src), bytes);
    std::unique_lock<std::mutex> lock(mutex_);
    work_done_.wait(lock, [this] { return pending_ == 0; });
  }

private:
  static void CopySlice(std::size_t index, std::size_t parts, std::uint8_t* dst,
                        const std::uint8_t* src, std::size_t bytes) noexcept {
    std::size_t slice = (bytes + parts - 1) / parts;
    slice = (slice + kSliceAlign - 1) / kSliceAlign * kSliceAlign;
    const std::size_t begin = std::min(bytes, index * slice);
    const std::size_t end = std::min(bytes, begin + slice);
    if (end > begin) {
      std::memcpy(dst + begin, src + begin, end - begin);
    }
  }

  void Work(std::size_t index) noexcept {
    std::uint64_t seen = 0;
    for (;;) {
      std::uint8_t* dst;
      const std::uint8_t* src;
      std::size_t bytes;
      std::size_t parts;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_ready_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) {
          return;
        }
        seen = generation_;
        dst = dst_;
        src = src_;
        bytes = bytes_;
        parts = active_;
      }
      CopySlice(index, parts, dst, src, bytes);
      bool last;
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        last = --pending_ == 0;
      }
      if (last) {
        work_done_.notify_one();
      }
    }
  }

  std::mutex caller_mutex_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  std::vector<std::thread> workers_;
  std::uint64_t generation_{0};
  std::uint8_t* dst_{nullptr};
  const std::uint8_t* src_{nullptr};
  std::size_t bytes_{0};
  std::size_t active_{0};
  std::size_t pending_{0};
  bool stop_{false};
};

}  // namespace gufo::models::ling_3_flash::rocm
