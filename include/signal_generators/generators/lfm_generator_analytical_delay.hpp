#pragma once

// ============================================================================
// LfmGeneratorAnalyticalDelay — LFM chirp со встроенной аналитической задержкой
//
// ЧТО:    LFM с per-antenna задержкой через подстановку времени:
//           s_delayed(t) = A · exp(j·(π·k·(t−τ)² + 2π·f_start·(t−τ))),
//         output = 0 при t < τ (сигнал ещё «не пришёл»). Задержка задаётся
//         вектором delay_us (микросекунды) — по одному значению на антенну.
//
// ЗАЧЕМ:  «Идеальная» задержка без интерполяционных артефактов — эталон
//         для тестов LchFarrow (polyphase fractional resampling) и
//         beamforming-пайплайнов: можно сравнивать LchFarrow-выход с
//         математически точным сдвигом. Не требует post-processing через
//         LchFarrow — задержка вшита в фазу.
//
// ПОЧЕМУ: - OpenCL-вариант (legacy). ROCm-вариант: LfmGeneratorAnalyticalDelayROCm.
//         - Move-only: cl_program/queue/context уникальны.
//         - backend не владеет — caller гарантирует переживание объекта.
//         - GenerateToCpu даёт 2D vector [antenna][sample] для удобства
//           проверки формы сигнала по каждой антенне.
//
// Использование:
//   signal_gen::LfmGeneratorAnalyticalDelay gen(backend, lfm_params);
//   gen.SetSampling(system);
//   gen.SetDelays({0.0f, 0.27f, 0.54f});  // микросекунды
//   auto out = gen.GenerateToGpu();
//   // out.data — cl_mem [antennas × points]; caller clReleaseMemObject(out.data).
//
// История:
//   - Создан: 2026-02-18 (legacy OpenCL-ветка)
// ============================================================================

#include <signal_generators/params/signal_request.hpp>
#include <signal_generators/params/system_sampling.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <cstdint>
#include <utility>

namespace signal_gen {

/**
 * @class LfmGeneratorAnalyticalDelay
 * @brief GPU/CPU LFM-генератор с аналитической per-antenna задержкой.
 *
 * @note Move-only: cl_program/queue/context уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note OpenCL-вариант. ROCm-аналог: LfmGeneratorAnalyticalDelayROCm.
 * @see signal_gen::LfmGeneratorAnalyticalDelayROCm
 *
 * @code
 * LfmGeneratorAnalyticalDelay gen(backend);
 * gen.SetParams(lfm_params);
 * gen.SetSampling(system);
 * gen.SetDelays({0.0f, 0.27f, 0.54f});  // microseconds
 *
 * // GPU
 * auto result = gen.GenerateToGpu();
 * clReleaseMemObject(result.data);
 *
 * // CPU
 * auto cpu = gen.GenerateToCpu();
 * @endcode
 */
class LfmGeneratorAnalyticalDelay {
public:
  /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
  using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

  LfmGeneratorAnalyticalDelay(drv_gpu_lib::IBackend* backend,
                               const LfmParams& params);
  ~LfmGeneratorAnalyticalDelay();

  // No copy
  LfmGeneratorAnalyticalDelay(const LfmGeneratorAnalyticalDelay&) = delete;
  LfmGeneratorAnalyticalDelay& operator=(const LfmGeneratorAnalyticalDelay&) = delete;

  // Move
  LfmGeneratorAnalyticalDelay(LfmGeneratorAnalyticalDelay&& other) noexcept;
  LfmGeneratorAnalyticalDelay& operator=(LfmGeneratorAnalyticalDelay&& other) noexcept;

  /// Set LFM parameters
  void SetParams(const LfmParams& params) { params_ = params; }
  const LfmParams& GetParams() const { return params_; }

  /// Set sampling parameters (fs, length)
  void SetSampling(const SystemSampling& system) { system_ = system; }
  const SystemSampling& GetSampling() const { return system_; }

  /**
   * @brief Set per-antenna delays in microseconds
   * @param delay_us Delays (float), one per antenna
   */
  void SetDelays(const std::vector<float>& delay_us);

  /**
   * @brief Generate on GPU
   * @return InputData<cl_mem> with [antennas * points] complex signal
   * @note Caller must release result.data via clReleaseMemObject()
   *   @test_check result.data != nullptr && result.antenna_count == delay_us_.size()
   */
  drv_gpu_lib::InputData<cl_mem> GenerateToGpu();

  /**
   * @brief Генерация на GPU с опциональным сбором событий профилирования.
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * Собирает события: "Kernel" (lfm_analytical_delay.cl)
   * @return InputData<cl_mem> [antennas × points × complex<float>]; caller обязан clReleaseMemObject result.data.
   *   @test_check result.data != nullptr
   */
  drv_gpu_lib::InputData<cl_mem> GenerateToGpu(ProfEvents* prof_events);

  /**
   * @brief Generate on CPU (reference)
   * @return [antenna][sample] complex<float>
   *   @test_check result.size() == delay_us_.size() && result[0].size() == system_.length
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

  uint32_t GetAntennas() const {
    return static_cast<uint32_t>(delay_us_.size());
  }
  const std::vector<float>& GetDelays() const { return delay_us_; }

private:
  void CompileKernel();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;
  LfmParams params_;
  SystemSampling system_;
  std::vector<float> delay_us_;

  // OpenCL
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_program program_ = nullptr;
};

} // namespace signal_gen
