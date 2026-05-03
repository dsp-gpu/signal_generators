#pragma once

// ============================================================================
// FormScriptGenerator — DSL-генератор сигналов с on-disk кэшем kernels (OpenCL)
//
// ЧТО:    Расширение FormSignalGenerator: преобразует FormParams в DSL-скрипт
//         (читаемый текстовый сегментный формат), генерирует OpenCL kernel
//         source с параметрами как #define, компилирует и кэширует на диск
//         (.cl source + .bin binary). manifest.json — индекс кернелов с
//         комментариями. Версионирование коллизий: name → name_00, name_01.
//
// ЗАЧЕМ:  Прогрев pipeline и воспроизводимость экспериментов: один раз
//         собрали kernel под конкретные FormParams (CW + LFM сегменты,
//         модуляции, паузы из script.json), сохранили на диск с
//         осмысленным именем — далее любой запуск вытаскивает binary
//         минуя hiprtc/clBuildProgram. Параметры как #define → компилятор
//         constant-folding'ит → быстрее runtime-параметров через kernel args.
//
// ПОЧЕМУ: - Только OpenCL-вариант (нет ENABLE_ROCM guard'а — файл не
//           собирается в ROCm-сборке через CMake). ROCm-аналог:
//           FormScriptGeneratorROCm (упрощённый, без on-disk binary cache).
//         - Move-only: cl_program/queue/context уникальны на инстанс.
//         - On-disk cache делегирован сервису core::KernelCacheService —
//           один источник правды для всех модулей (SRP).
//         - Два режима: SetParams→Compile→Generate ИЛИ LoadKernel→Generate.
//         - manifest.json — отдельная запись KernelManifestEntry с
//           комментарием, датой, params_string и backend ("opencl"/"rocm").
//
// Использование:
//   FormScriptGenerator gen(backend);
//   FormParams p; p.f0=1e6; p.antennas=8; p.points=4096;
//   gen.SetParams(p); gen.Compile();
//   auto input = gen.GenerateInputData();          // InputData<cl_mem>
//   gen.SaveKernel("my_signal", "CW 1MHz 8ch");    // .cl + .bin + manifest
//
//   // Повторный запуск — без recompile:
//   gen.LoadKernel("my_signal");
//   auto input2 = gen.GenerateInputData();
//
// История:
//   - Создан: 2026-02-17
// ============================================================================

#include <signal_generators/params/form_params.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <core/services/kernel_cache_service.hpp>

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <memory>
#include <utility>

namespace signal_gen {

/**
 * @struct KernelManifestEntry
 * @brief Запись в manifest.json об одном кернеле
 */
struct KernelManifestEntry {
  std::string name;           ///< Имя кернела (без расширения)
  std::string comment;        ///< Пользовательский комментарий
  std::string created;        ///< Дата создания ISO 8601
  std::string params_string;  ///< Строка параметров для восстановления
  std::string backend;        ///< "opencl" или "rocm"
};

/**
 * @class FormScriptGenerator
 * @brief DSL-генератор + on-disk кэш OpenCL kernels для формулы getX.
 *
 * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.
 * @note Только OpenCL. ROCm-аналог: FormScriptGeneratorROCm.
 * @see signal_gen::FormScriptGeneratorROCm
 * @see signal_gen::FormSignalGenerator
 */
class FormScriptGenerator {
public:
  /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
  using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

  explicit FormScriptGenerator(drv_gpu_lib::IBackend* backend);
  ~FormScriptGenerator();

  // No copy
  FormScriptGenerator(const FormScriptGenerator&) = delete;
  FormScriptGenerator& operator=(const FormScriptGenerator&) = delete;

  // Move
  FormScriptGenerator(FormScriptGenerator&& other) noexcept;
  FormScriptGenerator& operator=(FormScriptGenerator&& other) noexcept;

  // ══════════════════════════════════════════════════════════════════════
  // Параметры
  // ══════════════════════════════════════════════════════════════════════

  void SetParams(const FormParams& params);
  void SetParamsFromString(const std::string& params_str);
  const FormParams& GetParams() const { return params_; }

  // ══════════════════════════════════════════════════════════════════════
  // DSL / Kernel source
  // ══════════════════════════════════════════════════════════════════════

  /**
   * @brief Возвращает читаемый DSL-скрипт (текстовое представление getX-формулы по params_).
   *
   * @return Multi-line строка с DSL-описанием сигнала (Params/Defs/Signal секции).
   *   @test_check !result.empty()
   */
  std::string GenerateScript() const;

  /**
   * @brief Возвращает полный OpenCL kernel source (с PRNG-helpers и `#define`-параметрами).
   *
   * @return Готовый к clBuildProgram OpenCL C исходник.
   *   @test_check !result.empty()
   */
  std::string GenerateKernelSource() const;

  /**
   * @brief Компилирует OpenCL kernel из текущих params (внутри: GenerateKernelSource → clBuildProgram).
   */
  void Compile();

  // ══════════════════════════════════════════════════════════════════════
  // On-disk кэш кернелов
  // ══════════════════════════════════════════════════════════════════════

  /**
   * @brief Сохранить текущий kernel на диск
   * @param name Имя (без расширения): "work_sig0"
   * @param comment Комментарий (записывается в manifest)
   *
   * Создаёт: name.cl + bin/name_opencl.bin
   * При коллизии: старые файлы → name_00.cl, name_opencl_00.bin
   */
  void SaveKernel(const std::string& name, const std::string& comment = "");

  /**
   * @brief Загрузить kernel с диска по имени
   * @param name Имя кернела (без расширения)
   *
   * Приоритет: binary (fast) → source (compile + save binary)
   */
  void LoadKernel(const std::string& name);

  /**
   * @brief Возвращает список имён кернелов из on-disk manifest.json.
   *
   * @return vector<string> с именами (без расширений); пуст если кэш ещё не создан.
   *   @test_check result.size() >= 0 (количество ранее сохранённых через SaveKernel)
   */
  std::vector<std::string> ListKernels() const;

  // ══════════════════════════════════════════════════════════════════════
  // Генерация сигнала
  // ══════════════════════════════════════════════════════════════════════

  /**
   * @brief Генерация на GPU с метаданными (InputData — как в fft_func)
   * @return InputData<cl_mem> с data, antenna_count, n_point, gpu_memory_bytes
   * @note Вызывающий код должен освободить input.data через clReleaseMemObject()
   *   @test_check result.data != nullptr && result.antenna_count == params_.antennas
   */
  drv_gpu_lib::InputData<cl_mem> GenerateInputData();

  /**
   * @brief Генерация на GPU с опциональным сбором событий профилирования.
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr] }
   *
   * Собирает события: "Kernel" (form_script_signal kernel)
   * @return InputData<cl_mem> с data, antenna_count, n_point, gpu_memory_bytes; caller обязан clReleaseMemObject result.data.
   *   @test_check result.data != nullptr
   */
  drv_gpu_lib::InputData<cl_mem> GenerateInputData(ProfEvents* prof_events);

  /**
   * @brief Генерация с возвратом на CPU (по каналам)
   * @return vector[antenna_id][sample_id] complex<float>
   *   @test_check result.size() == params_.antennas && result[0].size() == params_.points
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

  // ══════════════════════════════════════════════════════════════════════
  // Info
  // ══════════════════════════════════════════════════════════════════════

  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  size_t GetTotalSamples() const {
    return static_cast<size_t>(params_.antennas) * params_.points;
  }

  bool IsReady() const { return program_ != nullptr; }
  const std::string& GetCurrentKernelSource() const { return kernel_source_; }

  /// Путь к директории кернелов
  static std::string GetKernelsDir();
  /// Путь к директории бинарников
  static std::string GetKernelsBinDir();

private:
  void CompileSource(const std::string& source);
  void ReleaseGpuResources();

  // OpenCL binary helpers (remain here — work with cl_program)
  std::vector<unsigned char> GetProgramBinary() const;
  void LoadFromBinary(const std::vector<unsigned char>& binary);
  void LoadFromSource(const std::string& source);

  std::string ParamsToString() const;

  drv_gpu_lib::IBackend* backend_ = nullptr;
  FormParams params_;
  std::string kernel_source_;
  std::string loaded_kernel_name_;  ///< имя загруженного с диска кернела

  /// On-disk kernel cache (delegated to DrvGPU service)
  std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;

  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_program program_ = nullptr;
};

} // namespace signal_gen
