#pragma once

// ============================================================================
// LfmGeneratorAnalyticalDelayROCm — LFM chirp с аналитической задержкой (ROCm)
//
// ЧТО:    ROCm/HIP-порт LfmGeneratorAnalyticalDelay. Та же формула:
//           s(t) = A · exp(j·(π·μ·(t−τ)² + 2π·f_start·(t−τ))),
//         где τ = delay_us[antenna] · 1e-6 (мкс → с), output = 0 при t < τ.
//         Kernel: generate_lfm_analytical_delay; компиляция через GpuContext.
//
// ЗАЧЕМ:  Эталон «идеальной» fractional-delay для тестирования LchFarrow
//         на ROCm-стеке (правило 09-rocm-only). Production-вариант для
//         multi-antenna симуляции в radar-pipeline на Linux + AMD.
//
// ПОЧЕМУ: - GpuContext (Ref03 Layer 1) → KernelCacheService один раз
//           компилирует HIP module, дальше hot-path без overhead.
//         - Move-only с default — GpuContext сам обрабатывает RAII.
//         - Stub-секция #else (Windows без ROCm) — все методы throw, чтобы
//           Python-биндинги собирались кросс-платформенно.
//
// Использование:
//   signal_gen::LfmGeneratorAnalyticalDelayROCm gen(rocm_backend, lfm_params);
//   gen.SetSampling(system);
//   gen.SetDelays({0.0f, 0.27f, 0.54f});  // микросекунды
//   auto out = gen.GenerateToGpu();
//   // out.data — void* (HIP device pointer); caller вызывает hipFree.
//
// История:
//   - Создан: 2026-03-22 (порт OpenCL-варианта на ROCm для main-ветки)
// ============================================================================

#if ENABLE_ROCM

#include <signal_generators/params/signal_request.hpp>
#include <signal_generators/params/system_sampling.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <core/interface/gpu_context.hpp>
#include <core/services/profiling_types.hpp>

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cstdint>

namespace signal_gen {

using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class LfmGeneratorAnalyticalDelayROCm
 * @brief ROCm/HIP LFM-генератор с аналитической per-antenna задержкой.
 *
 * @note Move-only: GPU-ресурсы (GpuContext, hipModule) уникальны.
 * @note Требует #if ENABLE_ROCM. На Windows — stub (все методы throw).
 * @note API совместим с LfmGeneratorAnalyticalDelay (OpenCL).
 * @see signal_gen::LfmGeneratorAnalyticalDelay (legacy OpenCL)
 * @see drv_gpu_lib::GpuContext (Layer 1 Ref03)
 */
class LfmGeneratorAnalyticalDelayROCm {
public:
  LfmGeneratorAnalyticalDelayROCm(drv_gpu_lib::IBackend* backend,
                                    const LfmParams& params);
  ~LfmGeneratorAnalyticalDelayROCm() = default;

  LfmGeneratorAnalyticalDelayROCm(const LfmGeneratorAnalyticalDelayROCm&) = delete;
  LfmGeneratorAnalyticalDelayROCm& operator=(const LfmGeneratorAnalyticalDelayROCm&) = delete;
  LfmGeneratorAnalyticalDelayROCm(LfmGeneratorAnalyticalDelayROCm&&) noexcept = default;
  LfmGeneratorAnalyticalDelayROCm& operator=(LfmGeneratorAnalyticalDelayROCm&&) noexcept = default;

  void SetParams(const LfmParams& params) { params_ = params; }
  void SetSampling(const SystemSampling& system) { system_ = system; }
  void SetDelays(const std::vector<float>& delay_us) { delay_us_ = delay_us; }

  const LfmParams& GetParams() const { return params_; }
  const SystemSampling& GetSampling() const { return system_; }
  const std::vector<float>& GetDelays() const { return delay_us_; }
  uint32_t GetAntennas() const { return static_cast<uint32_t>(delay_us_.size()); }

  /// GPU generate → InputData<void*> (caller must hipFree result.data)
  drv_gpu_lib::InputData<void*> GenerateToGpu(ROCmProfEvents* prof_events = nullptr);

  /// CPU reference
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  LfmParams params_;
  SystemSampling system_;
  std::vector<float> delay_us_;
  bool compiled_ = false;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

#include <signal_generators/params/signal_request.hpp>
#include <signal_generators/params/system_sampling.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <stdexcept>
#include <vector>
#include <complex>

namespace signal_gen {
class LfmGeneratorAnalyticalDelayROCm {
public:
  LfmGeneratorAnalyticalDelayROCm(drv_gpu_lib::IBackend*, const LfmParams&) {}
  void SetParams(const LfmParams&) {}
  void SetSampling(const SystemSampling&) {}
  void SetDelays(const std::vector<float>&) {}
  drv_gpu_lib::InputData<void*> GenerateToGpu(void* = nullptr) {
    throw std::runtime_error("LfmGeneratorAnalyticalDelayROCm: ROCm not enabled");
  }
  std::vector<std::vector<std::complex<float>>> GenerateToCpu() {
    throw std::runtime_error("LfmGeneratorAnalyticalDelayROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
