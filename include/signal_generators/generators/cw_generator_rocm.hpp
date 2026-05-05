#pragma once

// ============================================================================
// CwGeneratorROCm — генератор Continuous Wave (комплексной синусоиды) на ROCm/HIP
//
// ЧТО:    Порт CwGenerator на HIP runtime: s(t) = A·exp(j·(2π·f·t + φ)).
//         Для multi-beam режима частоты разнесены: f_i = f0 + i·freq_step.
//         Использует Ref03 GpuContext для компиляции ядра через hiprtc.
//
// ЗАЧЕМ:  CW — опорный тоновый сигнал для тестов FFT, оконных функций и
//         beamforming на целевой платформе DSP-GPU (Debian + ROCm 7.2+).
//         Один пик в спектре = один частотный bin → эталон для проверки
//         spectrum-модуля и калибровки тракта на AMD GPU.
//
// ПОЧЕМУ: - ROCm-вариант под `#if ENABLE_ROCM`. На Windows / без ROCm —
//           stub с throw std::runtime_error (не падает на этапе линковки).
//         - Move-only: GpuContext + compiled module уникальны на инстанс,
//           копирование = double-release HIP-ресурсов.
//         - kBlockSize = 256 — оптимум для warp=64 RDNA4 (gfx1201).
//         - backend не владеет (raw указатель) — DrvGPU создан выше по стеку.
//         - Lazy compile через EnsureCompiled() — первый вызов компилирует
//           kernel через hiprtc, далее переиспользуется.
//
// Использование:
//   signal_gen::CwGeneratorROCm gen(backend);
//   auto out = gen.GenerateToGpu(system, params, beam_count);
//   // out.data — void* HIP device pointer, caller вызывает hipFree(out.data)
//
// История:
//   - Создан: 2026-03-14
// ============================================================================

#if ENABLE_ROCM

#include <signal_generators/params/signal_request.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <core/interface/gpu_context.hpp>
#include <core/services/profiling_types.hpp>

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cstdint>

namespace signal_gen {

using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class CwGeneratorROCm
 * @brief ROCm/HIP-генератор CW (комплексной синусоиды) с поддержкой multi-beam.
 *
 * @ingroup grp_signal_generators
 * @note Move-only: GpuContext + compiled module уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note Доступен только при ENABLE_ROCM=1. OpenCL-вариант: CwGenerator.
 * @see signal_gen::CwGenerator
 */
class CwGeneratorROCm {
public:
  explicit CwGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~CwGeneratorROCm() = default;

  // Запрет копирования
  CwGeneratorROCm(const CwGeneratorROCm&) = delete;
  CwGeneratorROCm& operator=(const CwGeneratorROCm&) = delete;

  // Перемещение
  CwGeneratorROCm(CwGeneratorROCm&&) noexcept = default;
  CwGeneratorROCm& operator=(CwGeneratorROCm&&) noexcept = default;

  /**
   * @brief Генерация CW-сигнала на GPU
   * @param system     Параметры дискретизации (fs, length)
   * @param params     Параметры CW (f0, phase, amplitude, freq_step)
   *   @test_ref CwParams
   * @param beam_count Количество лучей
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   * @return InputData<void*> с GPU-сигналом (caller обязан hipFree result.data)
   *   @test_check result != nullptr
   */
  drv_gpu_lib::InputData<void*> GenerateToGpu(
      const SystemSampling& system,
      const CwParams& params,
      uint32_t beam_count,
      ROCmProfEvents* prof_events = nullptr);

  /**
   * @brief CPU reference генерация CW (для сверки с GPU). Multi-beam через freq_step.
   *
   * @param system Параметры дискретизации (fs, length).
   * @param params Параметры CW (f0, phase, amplitude, freq_step).
   *   @test_ref CwParams
   * @param beam_count Количество лучей в выходе.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   *
   * @return Массив [beam_count × system.length] complex<float> (interleaved beams).
   *   @test_check result.size() == beam_count * system.length
   */
  std::vector<std::complex<float>> GenerateToCpu(
      const SystemSampling& system,
      const CwParams& params,
      uint32_t beam_count);

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  bool compiled_ = false;
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

#include <signal_generators/params/signal_request.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <stdexcept>
#include <vector>
#include <complex>

namespace signal_gen {
class CwGeneratorROCm {
public:
  explicit CwGeneratorROCm(drv_gpu_lib::IBackend*) {}
  /**
   * @brief Stub: бросает runtime_error — GenerateToGpu доступен только в ROCm-сборке.
   *
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
  drv_gpu_lib::InputData<void*> GenerateToGpu(const SystemSampling&, const CwParams&, uint32_t, void* = nullptr) {
    throw std::runtime_error("CwGeneratorROCm: ROCm not enabled");
  }
  /**
   * @brief Stub: бросает runtime_error — GenerateToCpu доступен только в ROCm-сборке.
   *
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
  std::vector<std::complex<float>> GenerateToCpu(const SystemSampling&, const CwParams&, uint32_t) {
    throw std::runtime_error("CwGeneratorROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
