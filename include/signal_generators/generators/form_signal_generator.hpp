#pragma once

// ============================================================================
// FormSignalGenerator — мультиканальный GPU-генератор по формуле getX (OpenCL)
//
// ЧТО:    Параллельная генерация комплексного сигнала по антеннам:
//           X = a·norm·exp(j·(2π·f0·t + π·fdev/ti·(t-ti/2)² + φ))
//             + an·norm·(randn + j·randn)
//           X = 0 при t<0 или t>ti-dt
//         Поддержка: мультиканал, per-channel задержка (FIXED/LINEAR/RANDOM),
//         встроенный шум Philox-2x32 + Box-Muller, ЛЧМ при fdev≠0.
//
// ЗАЧЕМ:  Формирующий импульс radar-pipeline: один kernel запуск выдаёт
//         весь приёмный массив (antennas × points) с готовой задержкой и
//         шумом — без post-processing. Совместим с InputData<cl_mem> →
//         подаётся прямо в SpectrumMaximaFinder / FFTProcessor без копий.
//
// ПОЧЕМУ: - Это OpenCL-вариант под `#if !ENABLE_ROCM` (legacy nvidia-ветка).
//           ROCm-вариант: FormSignalGeneratorROCm.
//         - Standalone-класс (не наследует ISignalGenerator) — свой API
//           под мультиканальность и InputData<cl_mem> метаданные.
//         - Move-only: cl_program/queue/context уникальны на инстанс.
//         - backend не владеет (raw указатель) — DrvGPU выше по стеку.
//         - Шум встроен в kernel (Philox) — без отдельного NoiseGenerator
//           прохода → одна launch latency вместо двух.
//
// Использование:
//   FormSignalGenerator gen(backend);
//   FormParams p;
//   p.fs=12e6; p.f0=1e6; p.antennas=8; p.points=4096;
//   p.amplitude=1.0; p.noise_amplitude=0.1;
//   gen.SetParams(p);
//   auto input = gen.GenerateInputData();   // InputData<cl_mem>
//   // ... подать input в FFT / SpectrumMaximaFinder ...
//   clReleaseMemObject(input.data);
//
// История:
//   - Создан: 2026-02-17
// ============================================================================

#if !ENABLE_ROCM  // OpenCL-only: ROCm uses FormSignalGeneratorROCm

#include <signal_generators/params/form_params.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <string>
#include <utility>

namespace signal_gen {

/**
 * @class FormSignalGenerator
 * @brief OpenCL-генератор комплексных сигналов по формуле getX (мультиканал).
 *
 * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.
 * @note Доступен только в OpenCL-сборке. ROCm-вариант: FormSignalGeneratorROCm.
 * @see signal_gen::FormSignalGeneratorROCm
 * @see signal_gen::DelayedFormSignalGenerator
 */
class FormSignalGenerator {
public:
  /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
  using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

  explicit FormSignalGenerator(drv_gpu_lib::IBackend* backend);
  ~FormSignalGenerator();

  // No copy
  FormSignalGenerator(const FormSignalGenerator&) = delete;
  FormSignalGenerator& operator=(const FormSignalGenerator&) = delete;

  // Move
  FormSignalGenerator(FormSignalGenerator&& other) noexcept;
  FormSignalGenerator& operator=(FormSignalGenerator&& other) noexcept;

  /// Установить параметры
  void SetParams(const FormParams& params) { params_ = params; }

  /// Установить параметры из строки "f0=1e6,a=1.0,an=0.1,tau=0.001"
  void SetParamsFromString(const std::string& params_str) {
    params_ = FormParams::ParseFromString(params_str);
  }

  /**
   * @brief Генерация на GPU с метаданными (InputData — как в fft_func)
   * @return InputData<cl_mem> с data, antenna_count, n_point, gpu_memory_bytes
   * @note Вызывающий код должен освободить input.data через clReleaseMemObject()!
   */
  drv_gpu_lib::InputData<cl_mem> GenerateInputData();

  /**
   * @brief Генерация на GPU с опциональным сбором событий профилирования
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *
   * Собирает события: "Kernel" (form_signal.cl)
   */
  drv_gpu_lib::InputData<cl_mem> GenerateInputData(ProfEvents* prof_events);

  /**
   * @brief Генерация с возвратом на CPU (по каналам)
   * @return vector[antenna_id][sample_id] complex<float>
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

private:
  void CompileKernel();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;
  FormParams params_;

  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_program program_ = nullptr;
};

} // namespace signal_gen

#endif  // !ENABLE_ROCM
