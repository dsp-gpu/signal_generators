#pragma once

// ============================================================================
// LfmConjugateGenerator — комплексно-сопряжённый LFM (reference для dechirp)
//
// ЧТО:    Генерирует s_ref*(t) = exp(−j·(π·μ·t² + 2π·f_start·t)), где
//         μ = (f_end − f_start)/T. Это комплексно-сопряжённая копия LFM
//         при τ = 0. Производит один сигнал длиной system.length (без
//         multi-beam, без задержек).
//
// ЗАЧЕМ:  Опорный сигнал для dechirp-обработки (matched filter):
//           s_dc = s_rx(t) · s_ref*(t)  →  тон на f_beat = μ·τ.
//         После FFT каждый принятый эхо-сигнал даёт пик на частоте,
//         пропорциональной задержке (= дальности цели). Без conjugate
//         не получится pulse compression.
//
// ПОЧЕМУ: - OpenCL-вариант (legacy). ROCm-вариант: LfmConjugateGeneratorROCm.
//         - Move-only: cl_program/queue/context уникальны.
//         - backend не владеет — caller гарантирует переживание объекта.
//         - Аналог LfmGeneratorAnalyticalDelay, но (1) τ всегда 0,
//           (2) знак фазы инвертирован.
//
// Использование:
//   signal_gen::LfmConjugateGenerator gen(backend, lfm_params);
//   gen.SetSampling(system);
//   cl_mem ref = gen.GenerateToGpu();
//   // ... использовать в dechirp pipeline ...
//   clReleaseMemObject(ref);
//
// История:
//   - Создан: 2026-02-21 (legacy OpenCL-ветка)
// ============================================================================

#include <signal_generators/params/signal_request.hpp>
#include <signal_generators/params/system_sampling.hpp>
#include <core/interface/i_backend.hpp>

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <cstdint>
#include <utility>

namespace signal_gen {

/**
 * @class LfmConjugateGenerator
 * @brief GPU/CPU conjugate-LFM генератор — reference для dechirp.
 *
 * @note Move-only: cl_program/queue/context уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note OpenCL-вариант. ROCm-аналог: LfmConjugateGeneratorROCm.
 * @see signal_gen::LfmConjugateGeneratorROCm
 * @see HeterodyneDechirp (radar/heterodyne)
 *
 * @code
 * LfmConjugateGenerator gen(backend, lfm_params);
 * gen.SetSampling(system);
 *
 * // GPU
 * cl_mem ref = gen.GenerateToGpu();
 * // ... use in dechirp pipeline ...
 * clReleaseMemObject(ref);
 *
 * // CPU
 * auto ref_cpu = gen.GenerateToCpu();
 * @endcode
 */
class LfmConjugateGenerator {
public:
  /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
  using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

  LfmConjugateGenerator(drv_gpu_lib::IBackend* backend,
                         const LfmParams& params);
  ~LfmConjugateGenerator();

  // No copy
  LfmConjugateGenerator(const LfmConjugateGenerator&) = delete;
  LfmConjugateGenerator& operator=(const LfmConjugateGenerator&) = delete;

  // Move
  LfmConjugateGenerator(LfmConjugateGenerator&& other) noexcept;
  LfmConjugateGenerator& operator=(LfmConjugateGenerator&& other) noexcept;

  /// Set LFM parameters
  void SetParams(const LfmParams& params) { params_ = params; }
  const LfmParams& GetParams() const { return params_; }

  /// Set sampling parameters (fs, length)
  void SetSampling(const SystemSampling& system) { system_ = system; }
  const SystemSampling& GetSampling() const { return system_; }

  /**
   * @brief Generate conjugate LFM on GPU
   * @return cl_mem with [num_samples] complex signal (conj LFM)
   * @note Caller must release via clReleaseMemObject()
   *   @test_check result != nullptr (cl_mem [system_.length × complex<float>])
   */
  cl_mem GenerateToGpu();

  /**
   * @brief Генерация на GPU с опциональным сбором событий профилирования.
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * Собирает события: "Kernel" (lfm_conjugate.cl)
   * @return cl_mem [system_.length × complex<float>] с conj(LFM); caller обязан clReleaseMemObject.
   *   @test_check result != nullptr
   */
  cl_mem GenerateToGpu(ProfEvents* prof_events);

  /**
   * @brief Generate conjugate LFM on CPU (reference)
   * @return vector of complex<float>, length = system_.length
   *   @test_check result.size() == system_.length
   */
  std::vector<std::complex<float>> GenerateToCpu();

private:
  void CompileKernel();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;
  LfmParams params_;
  SystemSampling system_;

  // OpenCL
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_program program_ = nullptr;
};

} // namespace signal_gen