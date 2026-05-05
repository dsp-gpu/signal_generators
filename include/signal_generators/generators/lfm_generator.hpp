#pragma once

// ============================================================================
// LfmGenerator — генератор LFM (Linear Frequency Modulation, chirp) на OpenCL
//
// ЧТО:    Базовый chirp: s(t) = A · exp(j·(π·k·t² + 2π·f_start·t)), где
//         k = (f_end − f_start)/T — скорость свипа частоты. Реализует
//         ISignalGenerator (kind=LFM). GPU через OpenCL kernel + CPU-эталон.
//
// ЗАЧЕМ:  LFM — основной зондирующий сигнал radar (pulse compression). Для
//         согласованной фильтрации нужно генерировать chirp с известными
//         f_start, f_end, T_pulse, sample_rate и проверять корректность
//         сжатия импульса в spectrum/heterodyne модулях.
//
// ПОЧЕМУ: - OpenCL-вариант (legacy nvidia-ветка) под `#if !ENABLE_ROCM`.
//           ROCm-вариант: LfmGeneratorROCm.
//         - Move-only: cl_program/queue/context уникальны на инстанс
//           (копирование = double-release GPU-ресурсов).
//         - backend — raw указатель, не владеет: DrvGPU создан выше по стеку.
//         - GenerateToCpu — эталон для unit-тестов без GPU.
//
// Использование:
//   signal_gen::LfmGenerator gen(backend, LfmParams{
//       .f_start = 1e6f, .f_end = 5e6f, .duration = 100e-6f, .amplitude = 1.0f});
//   auto out = gen.GenerateToGpu(system, beam_count);
//   // out — cl_mem; caller вызывает clReleaseMemObject(out).
//
// История:
//   - Создан: 2026-02-13 (legacy OpenCL-ветка)
// ============================================================================

#include <signal_generators/i_signal_generator.hpp>
#include <core/interface/i_backend.hpp>

#include <CL/cl.h>
#include <cstring>
#include <utility>
#include <vector>

namespace signal_gen {

/**
 * @class LfmGenerator
 * @brief OpenCL-генератор LFM chirp с поддержкой multi-beam.
 *
 * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note Доступен только в OpenCL-сборке. ROCm-вариант: LfmGeneratorROCm.
 * @see signal_gen::LfmGeneratorROCm
 * @see signal_gen::ISignalGenerator
 */
class LfmGenerator : public ISignalGenerator {
public:
    /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
    using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

    LfmGenerator(drv_gpu_lib::IBackend* backend, const LfmParams& params);
    ~LfmGenerator() override;

    LfmGenerator(const LfmGenerator&) = delete;
    LfmGenerator& operator=(const LfmGenerator&) = delete;
    LfmGenerator(LfmGenerator&& other) noexcept;
    LfmGenerator& operator=(LfmGenerator&& other) noexcept;

    /**
     * @brief CPU reference генерация LFM chirp.
     *
     * @param system Параметры дискретизации (fs, length).
     * @param out Выходной буфер [out_size] complex<float>.
     * @param out_size Размер буфера (должен быть >= system.length).
     */
    void GenerateToCpu(const SystemSampling& system,
                       std::complex<float>* out, size_t out_size) override;

    /**
     * @brief GPU production генерация LFM chirp (multi-beam).
     *
     * @param system Параметры дискретизации (fs, length).
     * @param beam_count Количество лучей в выходе.
     *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
     *
     * @return cl_mem [beam_count × system.length × complex<float>]; caller обязан clReleaseMemObject.
     *   @test_check result != nullptr
     */
    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count = 1) override;

    /**
     * @brief Генерация на GPU с опциональным сбором событий профилирования.
     * @param prof_events nullptr → production (zero overhead); &vec → benchmark
     *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
     *
     * Собирает события: "Kernel" (lfm_kernel)
     * @param system Параметры дискретизации (fs, length).
     * @param beam_count Количество лучей в выходе.
     *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
     * @return cl_mem [beam_count × system.length × complex<float>]; caller обязан clReleaseMemObject.
     *   @test_check result != nullptr
     */
    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count,
                         ProfEvents* prof_events);

    /**
     * @brief Возвращает тип сигнала (для introspection).
     *
     * @return Всегда `SignalKind::LFM` для этого класса.
     *   @test_check result == SignalKind::LFM
     */
    SignalKind Kind() const override { return SignalKind::LFM; }

    void SetParams(const LfmParams& params) { params_ = params; }
    const LfmParams& GetParams() const { return params_; }

private:
    void CompileKernel();
    void ReleaseGpuResources();

    drv_gpu_lib::IBackend* backend_ = nullptr;
    LfmParams params_;

    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_program program_ = nullptr;
};

} // namespace signal_gen
