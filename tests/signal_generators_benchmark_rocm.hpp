#pragma once

// ============================================================================
// FormSignalGeneratorROCmBenchmark — бенчмарк GenerateInputData() (ROCm)
//
// ЧТО:    Наследник GpuBenchmarkBase: 5 warmup + 20 замерных прогонов
//         FormSignalGeneratorROCm::GenerateInputData().
// ЗАЧЕМ:  Измерить GPU-время генерации FormSignal через ProfilingFacade.
// ПОЧЕМУ: GpuBenchmarkBase инкапсулирует шаблон warmup/run/export,
//         чтобы не дублировать код в каждом модуле (правило 06-profiling).
//
// История: Создан: 2026-03-01
// ============================================================================

/**
 * @class FormSignalGeneratorROCmBenchmark
 * @brief Бенчмарк для FormSignalGeneratorROCm::GenerateInputData().
 * @note Не публичный API. Запускается через test_signal_generators_benchmark_rocm.hpp.
 */

#if ENABLE_ROCM

#include <signal_generators/generators/form_signal_generator_rocm.hpp>
#include <core/services/gpu_benchmark_base.hpp>
#include <core/services/profiling/profiling_facade.hpp>

#include <hip/hip_runtime.h>

namespace test_signal_generators_rocm {

// ─── Benchmark: FormSignalGeneratorROCm::GenerateInputData() ──────────────

class FormSignalGeneratorROCmBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (ROCm) — для GPUProfiler
   * @param gen     Ссылка на FormSignalGeneratorROCm (не владеет,
   *                SetParams() уже вызван)
   * @param cfg     Параметры бенчмарка
   */
  FormSignalGeneratorROCmBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::FormSignalGeneratorROCm& gen,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_FormSignalROCm"})
    : GpuBenchmarkBase(backend, "FormSignalROCm", cfg),
      gen_(gen) {}

protected:
  /// Warmup — GenerateInputData без timing (prof_events = nullptr)
  void ExecuteKernel() override {
    auto input = gen_.GenerateInputData();
    hipFree(input.data);
  }

  /// Замер — GenerateInputData с ROCmProfEvents → ProfilingFacade::BatchRecord
  void ExecuteKernelTimed() override {
    signal_gen::ROCmProfEvents events;
    auto input = gen_.GenerateInputData(&events);
    hipFree(input.data);
    drv_gpu_lib::profiling::ProfilingFacade::GetInstance()
        .BatchRecord(gpu_id_, "signal_generators/form_signal", events);
  }

private:
  signal_gen::FormSignalGeneratorROCm& gen_;
};

}  // namespace test_signal_generators_rocm

#endif  // ENABLE_ROCM
