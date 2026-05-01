#pragma once

// ============================================================================
// LfmConjugateGeneratorROCm — комплексно-сопряжённый LFM (ROCm/HIP)
//
// ЧТО:    ROCm/HIP-порт LfmConjugateGenerator. Та же формула:
//           s_ref*(t) = exp(−j·(π·μ·t² + 2π·f_start·t)),
//         где μ = (f_end − f_start)/T, T = num_samples / sample_rate.
//         Kernel: generate_lfm_conjugate (lfm_kernels_rocm.hpp);
//         компиляция через GpuContext (Ref03 Layer 1).
//
// ЗАЧЕМ:  HeterodyneDechirp использует этот сигнал как reference для
//           s_dc = s_rx(t) · s_ref*(t)  →  тон на f_beat = μ·τ.
//         Это ключевой шаг pulse compression в FMCW-радарах. На main-ветке
//         (Linux + AMD + ROCm 7.2+, правило 09-rocm-only) OpenCL не работает,
//         поэтому ROCm-вариант обязателен для production.
//
// ПОЧЕМУ: - GpuContext + KernelCacheService → hipModule компилируется один
//           раз, дальше hot-path без overhead.
//         - kBlockSize=256 — оптимум для warp=64 на RDNA4 (правило 13).
//         - Move-only с default — GpuContext сам обрабатывает RAII.
//         - Stub-секция #else (Windows без ROCm) — все методы throw, чтобы
//           Python-биндинги собирались кросс-платформенно.
//
// Использование:
//   signal_gen::LfmConjugateGeneratorROCm gen(rocm_backend, lfm_params);
//   gen.SetSampling(system);
//   void* ref = gen.GenerateToGpu();
//   // ... передать в HeterodyneDechirp ...
//   hipFree(ref);
//
// История:
//   - Создан: 2026-03-22 (порт OpenCL-варианта на ROCm для main-ветки)
// ============================================================================

#if ENABLE_ROCM

#include <signal_generators/params/signal_request.hpp>
#include <signal_generators/params/system_sampling.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/gpu_context.hpp>

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cstdint>

namespace signal_gen {

/**
 * @class LfmConjugateGeneratorROCm
 * @brief ROCm/HIP conjugate-LFM генератор для dechirp-обработки.
 *
 * @note Move-only: GPU-ресурсы (GpuContext, hipModule) уникальны.
 * @note Требует #if ENABLE_ROCM. На Windows — stub (все методы throw).
 * @note API совместим с LfmConjugateGenerator (OpenCL).
 * @see signal_gen::LfmConjugateGenerator (legacy OpenCL)
 * @see drv_gpu_lib::GpuContext (Layer 1 Ref03)
 */
class LfmConjugateGeneratorROCm {
public:
  explicit LfmConjugateGeneratorROCm(drv_gpu_lib::IBackend* backend,
                                      const LfmParams& params);
  ~LfmConjugateGeneratorROCm() = default;

  LfmConjugateGeneratorROCm(const LfmConjugateGeneratorROCm&) = delete;
  LfmConjugateGeneratorROCm& operator=(const LfmConjugateGeneratorROCm&) = delete;
  LfmConjugateGeneratorROCm(LfmConjugateGeneratorROCm&&) noexcept = default;
  LfmConjugateGeneratorROCm& operator=(LfmConjugateGeneratorROCm&&) noexcept = default;

  void SetParams(const LfmParams& params) { params_ = params; }
  const LfmParams& GetParams() const { return params_; }

  void SetSampling(const SystemSampling& system) { system_ = system; }
  const SystemSampling& GetSampling() const { return system_; }

  /**
   * @brief Generate conjugate LFM on GPU (ROCm)
   * @return HIP device pointer [num_samples × complex<float>]
   *         CALLER OWNS — must hipFree!
   */
  void* GenerateToGpu();

  /**
   * @brief Generate conjugate LFM on CPU (reference)
   * @return vector<complex<float>>, length = system_.length
   */
  std::vector<std::complex<float>> GenerateToCpu();

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  LfmParams params_;
  SystemSampling system_;
  bool compiled_ = false;
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM — Windows stub

#include <signal_generators/params/signal_request.hpp>
#include <signal_generators/params/system_sampling.hpp>
#include <core/interface/i_backend.hpp>
#include <stdexcept>
#include <vector>
#include <complex>

namespace signal_gen {

class LfmConjugateGeneratorROCm {
public:
  explicit LfmConjugateGeneratorROCm(drv_gpu_lib::IBackend*, const LfmParams&) {}
  void SetParams(const LfmParams&) {}
  void SetSampling(const SystemSampling&) {}
  void* GenerateToGpu() { throw std::runtime_error("LfmConjugateGeneratorROCm: ROCm not enabled"); }
  std::vector<std::complex<float>> GenerateToCpu() {
    throw std::runtime_error("LfmConjugateGeneratorROCm: ROCm not enabled");
  }
};

}  // namespace signal_gen

#endif  // ENABLE_ROCM
