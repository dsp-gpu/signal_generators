#pragma once

// ============================================================================
// DelayedFormSignalGeneratorROCm — getX + дробная задержка Farrow 48×5 (ROCm)
//
// ЧТО:    Композиция: FormSignalGeneratorROCm (чистый сигнал) +
//         LchFarrowROCm (дробная задержка Lagrange 48×5 per-antenna + шум).
//         ROCm-аналог DelayedFormSignalGenerator: тот же pipeline, HIP runtime.
//
// ЗАЧЕМ:  Эмуляция реального radar-приёма на целевой платформе DSP-GPU
//         (Debian + ROCm 7.2+, RX 9070 / MI100): разные задержки на
//         разные антенны (sub-sample точность) для тестов beamforming'а
//         и калибровки фазового центра.
//
// ПОЧЕМУ: - Под `#if ENABLE_ROCM`. На Windows / без ROCm — stub с throw.
//         - Композиция (не наследование) — SRP: координация здесь,
//           генерация в FormSignalGeneratorROCm, задержка/шум в LchFarrowROCm.
//         - Шум через LchFarrowROCm::SetNoise (ПОСЛЕ задержки) — иначе
//           шум интерполируется и теряет некоррелированность по выборкам.
//         - delay_us — МИКРОСЕКУНДЫ (контракт API со стороны Python).
//         - LoadMatrix(json_path) — опциональная загрузка матрицы Lagrange
//           48×5 из JSON; иначе используется встроенная.
//         - backend не владеет (raw указатель) — DrvGPU выше по стеку.
//
// Использование:
//   DelayedFormSignalGeneratorROCm gen(rocm_backend);
//   gen.SetParams(params);
//   gen.SetDelays({0.0f, 1.5f, 3.0f, /*...*/});   // мкс per-antenna
//   auto input = gen.GenerateInputData();          // InputData<void*>, hipFree
//   auto cpu = gen.GenerateToCpu();                // vector[antenna][sample]
//
// История:
//   - Создан: 2026-03-22
// ============================================================================

#if ENABLE_ROCM

#include <signal_generators/params/form_params.hpp>
#include <signal_generators/generators/form_signal_generator_rocm.hpp>
#include <spectrum/lch_farrow_rocm.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <string>

namespace signal_gen {

/**
 * @class DelayedFormSignalGeneratorROCm
 * @brief ROCm-композит: FormSignalGeneratorROCm + LchFarrowROCm (delay + noise).
 *
 * @ingroup grp_signal_generators
 * @note Доступен только при ENABLE_ROCM=1. OpenCL-вариант: DelayedFormSignalGenerator.
 * @note Шум идёт ПОСЛЕ задержки (через LchFarrowROCm::SetNoise), не до.
 * @see signal_gen::DelayedFormSignalGenerator
 * @see signal_gen::FormSignalGeneratorROCm
 * @see lch_farrow::LchFarrowROCm
 */
class DelayedFormSignalGeneratorROCm {
public:
  explicit DelayedFormSignalGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~DelayedFormSignalGeneratorROCm() = default;

  DelayedFormSignalGeneratorROCm(const DelayedFormSignalGeneratorROCm&) = delete;
  DelayedFormSignalGeneratorROCm& operator=(const DelayedFormSignalGeneratorROCm&) = delete;

  void SetParams(const FormParams& params);
  void SetDelays(const std::vector<float>& delay_us);
  /**
   * @brief Загружает матрицу Lagrange 48×5 для Farrow-фильтра из JSON-файла.
   *
   * @param json_path Путь к JSON с матрицей (формат: { "data": [[...], ...] }).
   *   @test { values=["/tmp/test_config.json"] }
   */
  void LoadMatrix(const std::string& json_path) { lch_farrow_.LoadMatrix(json_path); }

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  const std::vector<float>& GetDelays() const { return lch_farrow_.GetDelays(); }

  /**
   * @brief GPU production: чистый сигнал → задержка Farrow → шум. Возвращает InputData<void*>.
   *
   * @return InputData<void*> [antennas × points × complex<float>]; caller обязан hipFree result.data.
   *   @test_check result != nullptr && result.antenna_count == params_.antennas
   */
  drv_gpu_lib::InputData<void*> GenerateInputData();

  /**
   * @brief Полный pipeline с readback на CPU (для unit-тестов и сверки с GPU).
   *
   * @return vector[antenna_id][sample_id] complex<float>.
   *   @test_check result.size() == params_.antennas && result[0].size() == params_.points
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

private:
  drv_gpu_lib::IBackend* backend_;
  FormParams params_;
  FormSignalGeneratorROCm signal_gen_;
  lch_farrow::LchFarrowROCm lch_farrow_;
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
class DelayedFormSignalGeneratorROCm {
public:
  explicit DelayedFormSignalGeneratorROCm(drv_gpu_lib::IBackend*) {}
  void SetParams(const FormParams&) {}
  void SetDelays(const std::vector<float>&) {}
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
    throw std::runtime_error("DelayedFormSignalGeneratorROCm: ROCm not enabled");
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
    throw std::runtime_error("DelayedFormSignalGeneratorROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
