#pragma once

// ============================================================================
// NoiseGeneratorROCm — генератор гауссовского комплексного шума (ROCm/HIP)
//
// ЧТО:    ROCm/HIP-порт NoiseGenerator. Тот же алгоритм:
//         Philox-2x32-10 PRNG + Box-Muller для нормального распределения.
//         Через GpuContext (Ref03 Layer 1) и hipStream.
//
// ЗАЧЕМ:  Production-вариант шума для main-ветки (Linux + AMD + ROCm 7.2+,
//         правило 09-rocm-only). Используется в тестах SNR / детекторов /
//         radar-пайплайнов на реальном железе. Воспроизводимость по seed —
//         критична для регрессионных тестов.
//
// ПОЧЕМУ: - GpuContext + KernelCacheService → hipModule компилируется
//           один раз; hot-path без overhead.
//         - Philox-2x32 (counter-based) — каждая нить берёт уникальный
//           counter, параллелизм без коллизий PRNG-состояний.
//         - rng_ (std::mt19937) на хосте — для генерации seeds, если caller
//           не передал свои (не для основного шума, его делает GPU).
//         - kBlockSize=256 — оптимум для warp=64 на RDNA4 (правило 13).
//         - Stub-секция #else (Windows без ROCm) — все методы throw.
//
// Использование:
//   signal_gen::NoiseGeneratorROCm gen(rocm_backend);
//   auto out = gen.GenerateToGpu(system, NoiseParams{.amplitude=1.0f, .seed=42},
//                                 beam_count);
//   // out.data — void* (HIP device pointer); caller вызывает hipFree.
//
// История:
//   - Создан: 2026-03-14 (порт OpenCL-варианта на ROCm для main-ветки)
// ============================================================================

#if ENABLE_ROCM

#include <signal_generators/params/signal_request.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <core/interface/gpu_context.hpp>
#include <core/services/profiling_types.hpp>

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cstdint>
#include <random>

namespace signal_gen {

using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class NoiseGeneratorROCm
 * @brief ROCm/HIP-генератор гауссовского комплексного шума.
 *
 * @note Move-only: GPU-ресурсы (GpuContext, hipModule) уникальны.
 * @note Требует #if ENABLE_ROCM. На Windows — stub (все методы throw).
 * @note API совместим с NoiseGenerator (OpenCL) по семантике.
 * @see signal_gen::NoiseGenerator (legacy OpenCL)
 * @see drv_gpu_lib::GpuContext (Layer 1 Ref03)
 */
class NoiseGeneratorROCm {
public:
  explicit NoiseGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~NoiseGeneratorROCm() = default;

  NoiseGeneratorROCm(const NoiseGeneratorROCm&) = delete;
  NoiseGeneratorROCm& operator=(const NoiseGeneratorROCm&) = delete;
  NoiseGeneratorROCm(NoiseGeneratorROCm&&) noexcept = default;
  NoiseGeneratorROCm& operator=(NoiseGeneratorROCm&&) noexcept = default;

  drv_gpu_lib::InputData<void*> GenerateToGpu(
      const SystemSampling& system,
      const NoiseParams& params,
      uint32_t beam_count,
      ROCmProfEvents* prof_events = nullptr);

  std::vector<std::complex<float>> GenerateToCpu(
      const SystemSampling& system,
      const NoiseParams& params,
      uint32_t beam_count);

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  bool compiled_ = false;
  std::mt19937 rng_{std::random_device{}()};
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

#include <signal_generators/params/signal_request.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <stdexcept>
#include <vector>
#include <complex>

namespace signal_gen {
class NoiseGeneratorROCm {
public:
  explicit NoiseGeneratorROCm(drv_gpu_lib::IBackend*) {}
  drv_gpu_lib::InputData<void*> GenerateToGpu(const SystemSampling&, const NoiseParams&, uint32_t, void* = nullptr) {
    throw std::runtime_error("NoiseGeneratorROCm: ROCm not enabled");
  }
  std::vector<std::complex<float>> GenerateToCpu(const SystemSampling&, const NoiseParams&, uint32_t) {
    throw std::runtime_error("NoiseGeneratorROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
