#pragma once

// ============================================================================
// DelayedFormSignalGenerator — getX + дробная задержка Farrow 48×5 (OpenCL)
//
// ЧТО:    Композиция: FormSignalGenerator (чистый сигнал, noise=0) + LchFarrow
//         (дробная задержка Lagrange 48×5 per-antenna + шум через SetNoise).
//         Задержка задаётся в микросекундах на каждую антенну, шум
//         добавляется ПОСЛЕ применения задержки.
//
// ЗАЧЕМ:  Эмуляция реального radar-приёма: разные пути сигнала к разным
//         антеннам = разные задержки (sub-sample точность). Альтернатива
//         post-processing'у через LchFarrow в pipeline'е — здесь delay
//         встроен в этап генерации, упрощает тесты beamforming'а и
//         калибровки фазового центра.
//
// ПОЧЕМУ: - Только OpenCL (`#if !ENABLE_ROCM`). ROCm-аналог:
//           DelayedFormSignalGeneratorROCm.
//         - Композиция (не наследование) — SRP: этот класс координирует,
//           FormSignalGenerator генерирует, LchFarrow задерживает/шумит.
//         - Move-only: содержит move-only компоненты.
//         - Шум через LchFarrow (а не через FormSignalGenerator): шум
//           идёт ПОСЛЕ задержки — иначе шум тоже бы интерполировался,
//           что меняет его статистику (некоррелированность по выборкам).
//         - delay_us — МИКРОСЕКУНДЫ (контракт API со стороны Python).
//         - Матрица Lagrange 48×5: 48 фаз × 5 коэффициентов на фазу.
//           Опционально загружается из JSON (LoadMatrix), иначе встроенная.
//
// Использование:
//   DelayedFormSignalGenerator gen(backend);
//   FormParams p; p.fs=12e6; p.f0=1e6; p.antennas=8; p.points=4096;
//   p.amplitude=1.0; p.noise_amplitude=0.1;
//   gen.SetParams(p);
//   gen.SetDelays({0.0f, 1.5f, 3.0f, 4.5f, 6.0f, 7.5f, 9.0f, 10.5f}); // мкс
//   auto input = gen.GenerateInputData();
//   clReleaseMemObject(input.data);
//
// История:
//   - Создан: 2026-02-17
// ============================================================================

#if !ENABLE_ROCM  // OpenCL-only: ROCm uses DelayedFormSignalGeneratorROCm

#include <signal_generators/params/form_params.hpp>
#include <signal_generators/generators/form_signal_generator.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <spectrum/lch_farrow.hpp>

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <string>
#include <utility>

namespace signal_gen {

/**
 * @class DelayedFormSignalGenerator
 * @brief OpenCL-генератор getX с дробной задержкой Farrow (Lagrange 48×5).
 *
 * @note Move-only (composite над move-only компонентами).
 * @note Шум идёт ПОСЛЕ задержки (через LchFarrow::SetNoise), не до.
 * @note Только OpenCL. ROCm-аналог: DelayedFormSignalGeneratorROCm.
 * @see signal_gen::DelayedFormSignalGeneratorROCm
 * @see signal_gen::FormSignalGenerator
 * @see lch_farrow::LchFarrow
 */
class DelayedFormSignalGenerator {
public:
  /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
  using ProfEvents = lch_farrow::ProfEvents;

  explicit DelayedFormSignalGenerator(drv_gpu_lib::IBackend* backend);
  ~DelayedFormSignalGenerator() = default;

  // No copy
  DelayedFormSignalGenerator(const DelayedFormSignalGenerator&) = delete;
  DelayedFormSignalGenerator& operator=(const DelayedFormSignalGenerator&) = delete;

  // Move
  DelayedFormSignalGenerator(DelayedFormSignalGenerator&&) = default;
  DelayedFormSignalGenerator& operator=(DelayedFormSignalGenerator&&) = default;

  /// Установить параметры сигнала (FormParams)
  void SetParams(const FormParams& params);

  /// Установить параметры из строки
  void SetParamsFromString(const std::string& params_str);

  /**
   * @brief Установить задержки per-antenna в микросекундах
   * @param delay_us Вектор задержек (float), delay_us.size() == antennas
   */
  void SetDelays(const std::vector<float>& delay_us);

  /**
   * @brief Загрузить матрицу Lagrange из JSON-файла
   * @param json_path Путь к файлу (формат: { "data": [[...], ...] })
   *   @test { values=["/tmp/test_config.json"] }
   *
   * Если не вызвано — используется встроенная матрица 48×5 из LchFarrow.
   */
  void LoadMatrix(const std::string& json_path);

  /**
   * @brief Генерация на GPU: сигнал + задержка + шум
   * @return InputData<cl_mem>
   * @note Вызывающий код освобождает result.data через clReleaseMemObject()
   *   @test_check result.data != nullptr && result.antenna_count == params.antennas
   */
  drv_gpu_lib::InputData<cl_mem> GenerateInputData();

  /**
   * @brief Генерация на GPU с опциональным сбором событий профилирования.
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * Собирает события: "Kernel" (FormSignal), "Upload_delay", "Kernel" (FarrowDelay)
   * @return InputData<cl_mem> с задержанным FormSignal; caller обязан clReleaseMemObject result.data.
   *   @test_check result.data != nullptr && result.antenna_count == params.antennas
   */
  drv_gpu_lib::InputData<cl_mem> GenerateInputData(ProfEvents* prof_events);

  /**
   * @brief Генерация с возвратом на CPU
   * @return vector[antenna_id][sample_id] complex<float>
   *   @test_check result.size() == params.antennas && result[0].size() == params.points
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

  // ══════════════════════════════════════════════════════════════════════
  // Getters
  // ══════════════════════════════════════════════════════════════════════

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  size_t GetTotalSamples() const {
    return static_cast<size_t>(params_.antennas) * params_.points;
  }
  const std::vector<float>& GetDelays() const { return lch_farrow_.GetDelays(); }

private:
  drv_gpu_lib::IBackend* backend_ = nullptr;
  FormParams params_;

  FormSignalGenerator signal_gen_;
  lch_farrow::LchFarrow lch_farrow_;
};

} // namespace signal_gen

#endif  // !ENABLE_ROCM
