#pragma once

// ============================================================================
// FormScriptGeneratorROCm — ROCm-обёртка: FormParams → DSL → HIP signal
//
// ЧТО:    Упрощённый ROCm-порт FormScriptGenerator. Преобразует FormParams
//         в DSL-текст (BuildScript), затем делегирует компиляцию и запуск
//         базовому ScriptGeneratorROCm (через hiprtc). Без on-disk binary
//         cache (в HIP бинарники привязаны к gfx-арке, кэширование делает
//         сам hiprtc/JIT).
//
// ЗАЧЕМ:  Единый API для генерации сигналов из FormParams независимо от
//         backend'а: тот же FormParams → одинаковый сигнал на OpenCL и
//         ROCm. Удобно в radar-pipeline и тестах: SetParams → Generate,
//         не зная деталей платформы.
//
// ПОЧЕМУ: - Под `#if ENABLE_ROCM`. На Windows / без ROCm — stub с throw.
//         - Композиция (не наследование) над ScriptGeneratorROCm: SRP —
//           этот класс отвечает только за FormParams→DSL, компиляцию и
//           launch делает базовый класс.
//         - backend не владеет (raw указатель) — DrvGPU выше по стеку.
//         - Без on-disk binary cache (ROCm hiprtc сам кэширует JIT).
//
// Использование:
//   FormScriptGeneratorROCm gen(rocm_backend);
//   gen.SetParams(params);
//   auto input = gen.GenerateInputData();   // InputData<void*>, hipFree caller
//   auto cpu_data = gen.GenerateToCpu();
//   std::string hip_src = gen.GetKernelSource();   // для дебага
//
// История:
//   - Создан: 2026-03-22
// ============================================================================

#if ENABLE_ROCM

#include <signal_generators/params/form_params.hpp>
#include <signal_generators/generators/script_generator_rocm.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <string>

namespace signal_gen {

/**
 * @class FormScriptGeneratorROCm
 * @brief Композиция над ScriptGeneratorROCm: FormParams → DSL → HIP-сигнал.
 *
 * @ingroup grp_signal_generators
 * @note Доступен только при ENABLE_ROCM=1. OpenCL-вариант: FormScriptGenerator.
 * @see signal_gen::FormScriptGenerator
 * @see signal_gen::ScriptGeneratorROCm
 */
class FormScriptGeneratorROCm {
public:
  explicit FormScriptGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~FormScriptGeneratorROCm() = default;

  void SetParams(const FormParams& params);
  void SetParamsFromString(const std::string& params_str);

  /**
   * @brief GPU production: FormParams → DSL → hiprtc → HIP launch. Возвращает InputData<void*>.
   *
   * @return InputData<void*> [antennas × points × complex<float>]; caller обязан hipFree result.data.
   *   @test_check result != nullptr && result.antenna_count == params_.antennas
   */
  drv_gpu_lib::InputData<void*> GenerateInputData();

  /**
   * @brief Полный pipeline с readback на CPU (для unit-тестов и сверки).
   *
   * @return vector[antenna_id][sample_id] complex<float>.
   *   @test_check result.size() == params_.antennas && result[0].size() == params_.points
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }

  /// Сгенерированный HIP kernel source (для отладки)
  const std::string& GetKernelSource() const { return script_gen_.GetKernelSource(); }

private:
  /// Построить текст DSL-скрипта из FormParams
  std::string BuildScript() const;

  drv_gpu_lib::IBackend* backend_;
  FormParams params_;
  ScriptGeneratorROCm script_gen_;
  bool compiled_ = false;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

#include <signal_generators/params/form_params.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <stdexcept>
#include <vector>
#include <complex>

namespace signal_gen {
class FormScriptGeneratorROCm {
public:
  explicit FormScriptGeneratorROCm(drv_gpu_lib::IBackend*) {}
  void SetParams(const FormParams&) {}
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
    throw std::runtime_error("FormScriptGeneratorROCm: ROCm not enabled");
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
    throw std::runtime_error("FormScriptGeneratorROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
