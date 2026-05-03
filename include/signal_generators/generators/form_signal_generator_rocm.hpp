#pragma once

// ============================================================================
// FormSignalGeneratorROCm — мультиканальный генератор по формуле getX (ROCm/HIP)
//
// ЧТO:    Порт FormSignalGenerator на HIP runtime: тот же алгоритм getX
//         (CW + ЛЧМ + Philox-шум, один kernel на весь массив antennas×points),
//         но с hiprtc для компиляции, void* device pointers вместо cl_mem,
//         hipStream_t вместо cl_command_queue. Использует Ref03 GpuContext.
//
// ЗАЧЕМ:  Целевая платформа DSP-GPU — Debian + ROCm 7.2+ (RX 9070, MI100).
//         OpenCL-ветка legacy; формирование импульса radar-pipeline на
//         AMD GPU делает этот класс. Один launch → готовый InputData<void*>
//         для FFTProcessor / SpectrumMaximaFinder без копий.
//
// ПОЧЕМУ: - Под `#if ENABLE_ROCM`. На Windows / без ROCm — stub с throw,
//           чтобы линковка не падала, а вызов давал понятный runtime error.
//         - Move-only: GpuContext + compiled module уникальны на инстанс.
//         - kBlockSize = 256 — оптимум warp=64 на RDNA4/CDNA1.
//         - Шум встроен в kernel (Philox-2x32 + Box-Muller) — одна launch
//           latency вместо двух (без отдельного NoiseGenerator прохода).
//         - backend не владеет (raw указатель) — DrvGPU выше по стеку.
//
// Использование:
//   FormSignalGeneratorROCm gen(rocm_backend);
//   gen.SetParams(params);
//   auto result = gen.GenerateInputData();
//   // result.data — void* (HIP device pointer), caller обязан hipFree()
//   auto cpu_data = gen.GenerateToCpu();   // vector[antenna][sample]
//
// История:
//   - Создан: 2026-02-23
// ============================================================================

#if ENABLE_ROCM

#include <signal_generators/params/form_params.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <core/interface/gpu_context.hpp>
#include <core/services/profiling_types.hpp>

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <utility>

namespace signal_gen {

/// Список событий профилирования ROCm (имя стадии + ROCmProfilingData)
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class FormSignalGeneratorROCm
 * @brief ROCm/HIP-генератор getX (мультиканал, встроенный Philox-шум, ЛЧМ).
 *
 * @ingroup grp_signal_generators
 * @note Move-only: GpuContext + compiled module уникальны на инстанс.
 * @note Доступен только при ENABLE_ROCM=1. OpenCL-вариант: FormSignalGenerator.
 * @see signal_gen::FormSignalGenerator
 * @see signal_gen::DelayedFormSignalGeneratorROCm
 */
class FormSignalGeneratorROCm {
public:
  explicit FormSignalGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~FormSignalGeneratorROCm();

  // No copy
  FormSignalGeneratorROCm(const FormSignalGeneratorROCm&) = delete;
  FormSignalGeneratorROCm& operator=(const FormSignalGeneratorROCm&) = delete;

  // Move
  FormSignalGeneratorROCm(FormSignalGeneratorROCm&& other) noexcept;
  FormSignalGeneratorROCm& operator=(FormSignalGeneratorROCm&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  void SetParams(const FormParams& params) { params_ = params; }
  void SetParamsFromString(const std::string& params_str) {
    params_ = FormParams::ParseFromString(params_str);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Processing
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Генерация сигнала на GPU (ROCm)
   * @return InputData<void*> с сгенерированным сигналом (caller обязан hipFree result.data)
   *   @test_check result != nullptr
   */
  drv_gpu_lib::InputData<void*> GenerateInputData();

  /**
   * @brief Генерация на GPU с опциональным сбором событий профилирования (ROCm)
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr] }
   *
   * Собирает события: "Kernel" (generate_form_signal HIP kernel)
   * @return InputData<void*> с HIP device pointer; caller обязан hipFree result.data.
   *   @test_check result != nullptr
   */
  drv_gpu_lib::InputData<void*> GenerateInputData(ROCmProfEvents* prof_events);

  /**
   * @brief Генерация на GPU с возвратом результата на CPU
   * @return vector[antenna][sample] = complex<float>
   *   @test_check result.size() == params_.antennas && result[0].size() == params_.points
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

  // ════════════════════════════════════════════════════════════════════════
  // Getters
  // ════════════════════════════════════════════════════════════════════════

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  size_t GetTotalSamples() const {
    return static_cast<size_t>(params_.antennas) * params_.points;
  }

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  FormParams params_;
  bool compiled_ = false;

  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

// ═══════════════════════════════════════════════════════════════════════════
// Stub for non-ROCm builds (Windows)
// ═══════════════════════════════════════════════════════════════════════════

#include <signal_generators/params/form_params.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>

#include <stdexcept>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace signal_gen {

class FormSignalGeneratorROCm {
public:
  explicit FormSignalGeneratorROCm(drv_gpu_lib::IBackend*) {}
  ~FormSignalGeneratorROCm() = default;

  void SetParams(const FormParams& params) { params_ = params; }
  void SetParamsFromString(const std::string& params_str) {
    params_ = FormParams::ParseFromString(params_str);
  }

  /**
   * @brief Stub: бросает runtime_error — GenerateInputData доступен только в ROCm-сборке.
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
  drv_gpu_lib::InputData<void*> GenerateInputData() {
    throw std::runtime_error("FormSignalGeneratorROCm: ROCm not enabled");
  }

  /**
   * @brief Stub: бросает runtime_error — GenerateToCpu доступен только в ROCm-сборке.
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu() {
    throw std::runtime_error("FormSignalGeneratorROCm: ROCm not enabled");
  }

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  size_t GetTotalSamples() const {
    return static_cast<size_t>(params_.antennas) * params_.points;
  }

private:
  FormParams params_;
};

}  // namespace signal_gen

#endif  // ENABLE_ROCM
