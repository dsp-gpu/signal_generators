#pragma once

// ============================================================================
// ScriptGeneratorROCm — компилятор DSL в HIP kernel через GpuContext (disk cache)
//
// ЧТО:    ROCm/HIP-порт ScriptGenerator. Тот же DSL-формат
//         ([Params]/[Defs]/[Signal]), но трансляция:
//         DSL → HIP C++ kernel source → GpuContext::CompileModule
//         → disk cache (kernel_cache_v2, ключ — CompileKey.Hash).
//
// ЗАЧЕМ:  Production-вариант runtime-компилируемых сигналов на main-ветке
//         (Linux + AMD + ROCm 7.2+, правило 09-rocm-only). Главный плюс
//         перед OpenCL: HSACO-кеш на диске — повторный запуск с тем же
//         скриптом не перекомпилирует HIP-модуль (минус ~сотни мс старта).
//
// ПОЧЕМУ: - GpuContext per-script (не shared!) — CompileModule идемпотентен
//           один раз на контекст; разные пользовательские скрипты =
//           разный CompileKey.Hash = разные HSACO-файлы (disk cache работает
//           через имя файла, без коллизий между скриптами).
//         - kernel_fn_ кеширует hipFunction_t после CompileModule, чтобы
//           Generate() не лез в KernelCacheService на каждый вызов.
//         - Move-only: GpuContext (unique_ptr) + hipStream уникальны.
//         - Stub-секция #else (Windows без ROCm) — все методы throw.
//
// Использование:
//   signal_gen::ScriptGeneratorROCm gen(rocm_backend);
//   gen.LoadFile("scripts/my_signal.signal");
//   void* out = gen.Generate();
//   // ... использовать ...
//   hipFree(out);
//
// История:
//   - Создан:    2026-03-22 (порт OpenCL-варианта на ROCm)
//   - Обновлён:  2026-04-22 (миграция на kernel_cache_v2 / CompileKey.Hash)
// ============================================================================

#if ENABLE_ROCM

#include <core/interface/i_backend.hpp>
#include <core/interface/gpu_context.hpp>

#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <cstdint>

namespace signal_gen {

struct ScriptParams;   // forward (defined in script_generator.hpp)
struct ParsedScript;   // forward

/**
 * @class ScriptGeneratorROCm
 * @brief ROCm/HIP DSL-генератор с disk-cache HSACO через GpuContext.
 *
 * @note Move-only: GpuContext (unique_ptr) и hipStream уникальны.
 * @note Требует #if ENABLE_ROCM. На Windows — stub (все методы throw).
 * @note Per-script GpuContext: каждый LoadScript пересоздаёт контекст
 *       (см. ctx_) — disk cache ключуется CompileKey.Hash.
 * @see signal_gen::ScriptGenerator (legacy OpenCL)
 * @see drv_gpu_lib::GpuContext (Layer 1 Ref03)
 */
class ScriptGeneratorROCm {
public:
  explicit ScriptGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~ScriptGeneratorROCm();

  ScriptGeneratorROCm(const ScriptGeneratorROCm&) = delete;
  ScriptGeneratorROCm& operator=(const ScriptGeneratorROCm&) = delete;
  ScriptGeneratorROCm(ScriptGeneratorROCm&& other) noexcept;
  ScriptGeneratorROCm& operator=(ScriptGeneratorROCm&& other) noexcept;

  void LoadScript(const std::string& script_text);
  void LoadFile(const std::string& file_path);

  /// Generate on GPU → void* (caller must hipFree)
  void* Generate();

  /// Generate and read back to CPU
  std::vector<std::complex<float>> GenerateToCpu();

  uint32_t GetAntennas() const;
  uint32_t GetPoints() const;
  size_t GetTotalSamples() const;
  const std::string& GetKernelSource() const { return kernel_source_; }
  bool IsReady() const { return kernel_fn_ != nullptr; }

private:
  ParsedScript ParseScript(const std::string& text);
  std::string GenerateHIPKernelSource(const ParsedScript& script);
  std::string PrepareExpression(const std::string& line);
  void CompileKernel(const std::string& source);

  static std::string Trim(const std::string& s);
  static std::string ToUpper(const std::string& s);

  drv_gpu_lib::IBackend* backend_ = nullptr;
  hipStream_t stream_ = nullptr;

  /// Per-script GpuContext — recreated on each LoadScript because CompileModule
  /// is idempotent (one source per context). Different user scripts → unique
  /// CompileKey.Hash → unique HSACO file, так что disk cache работает через имя файла.
  std::unique_ptr<drv_gpu_lib::GpuContext> ctx_;

  /// Cached kernel function pointer after ctx_->CompileModule().
  hipFunction_t kernel_fn_ = nullptr;

  uint32_t antennas_ = 0;
  uint32_t points_ = 0;
  std::string kernel_source_;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

#include <core/interface/i_backend.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <complex>

namespace signal_gen {
class ScriptGeneratorROCm {
public:
  explicit ScriptGeneratorROCm(drv_gpu_lib::IBackend*) {}
  void LoadScript(const std::string&) { throw std::runtime_error("ScriptGeneratorROCm: ROCm not enabled"); }
  void* Generate() { throw std::runtime_error("ScriptGeneratorROCm: ROCm not enabled"); }
  std::vector<std::complex<float>> GenerateToCpu() { throw std::runtime_error("ScriptGeneratorROCm: ROCm not enabled"); }
  bool IsReady() const { return false; }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
