#pragma once

// ============================================================================
// LfmGeneratorROCm — генератор LFM chirp (ROCm/HIP)
//
// ЧТО:    ROCm-порт LfmGenerator. Та же формула chirp:
//         s(t) = A · exp(j·(π·k·t² + 2π·f_start·t)),
//         но через HIP runtime + GpuContext (Ref03 Layer 1) — hipModule,
//         hipStream, void* device pointers. Возвращает InputData<void*>.
//
// ЗАЧЕМ:  Main-ветка DSP-GPU работает на Linux + AMD + ROCm 7.2+ (правило
//         09-rocm-only). LfmGenerator (OpenCL) — legacy nvidia, не работает
//         на RDNA4. LfmGeneratorROCm — production-вариант для радара.
//
// ПОЧЕМУ: - GpuContext + KernelCacheService → hipModuleLoad один раз,
//           дальше hot-path без overhead перекомпиляции.
//         - kBlockSize=256 — оптимум для warp=64 на RDNA4 (правило 13).
//         - Stub-секция #else (Windows без ROCm) — все методы throw, чтобы
//           Python-биндинги собирались кросс-платформенно (одна сборка).
//         - ROCmProfEvents — лист пар (имя, ROCmProfilingData), nullptr →
//           production без overhead, &vec → benchmark с замерами hipEvent.
//
// Использование:
//   signal_gen::LfmGeneratorROCm gen(rocm_backend);
//   auto out = gen.GenerateToGpu(system, lfm_params, beam_count);
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

namespace signal_gen {

using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class LfmGeneratorROCm
 * @brief ROCm/HIP-генератор LFM chirp с multi-beam.
 *
 * @note Move-only: GPU-ресурсы (GpuContext, hipModule) уникальны.
 * @note Требует #if ENABLE_ROCM. На Windows — stub (все методы throw).
 * @note API совместим с LfmGenerator для прозрачной замены backend'а.
 * @see signal_gen::LfmGenerator (legacy OpenCL)
 * @see drv_gpu_lib::GpuContext (Layer 1 Ref03)
 */
class LfmGeneratorROCm {
public:
  explicit LfmGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~LfmGeneratorROCm() = default;

  LfmGeneratorROCm(const LfmGeneratorROCm&) = delete;
  LfmGeneratorROCm& operator=(const LfmGeneratorROCm&) = delete;
  LfmGeneratorROCm(LfmGeneratorROCm&&) noexcept = default;
  LfmGeneratorROCm& operator=(LfmGeneratorROCm&&) noexcept = default;

  drv_gpu_lib::InputData<void*> GenerateToGpu(
      const SystemSampling& system,
      const LfmParams& params,
      uint32_t beam_count,
      ROCmProfEvents* prof_events = nullptr);

  std::vector<std::complex<float>> GenerateToCpu(
      const SystemSampling& system,
      const LfmParams& params,
      uint32_t beam_count);

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  bool compiled_ = false;
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
class LfmGeneratorROCm {
public:
  explicit LfmGeneratorROCm(drv_gpu_lib::IBackend*) {}
  drv_gpu_lib::InputData<void*> GenerateToGpu(const SystemSampling&, const LfmParams&, uint32_t, void* = nullptr) {
    throw std::runtime_error("LfmGeneratorROCm: ROCm not enabled");
  }
  std::vector<std::complex<float>> GenerateToCpu(const SystemSampling&, const LfmParams&, uint32_t) {
    throw std::runtime_error("LfmGeneratorROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
