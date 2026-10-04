// include/strata/platform/cpu_relax.hpp - the spin-wait hint and the store fence, per CPU architecture.
// x86: `pause` / `sfence`.  Arm64: `yield` / `dmb ishst`.
#pragma once

#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
namespace strata {
inline void cpu_pause() noexcept { _mm_pause(); }
inline void store_fence() noexcept { _mm_sfence(); }
}  // namespace strata
#elif defined(__aarch64__)
namespace strata {
inline void cpu_pause() noexcept { __asm__ volatile("yield" ::: "memory"); }
inline void store_fence() noexcept { __asm__ volatile("dmb ishst" ::: "memory"); }
}  // namespace strata
#else
#include <atomic>
namespace strata {
inline void cpu_pause() noexcept {}
inline void store_fence() noexcept { std::atomic_thread_fence(std::memory_order_release); }
}  // namespace strata
#endif
