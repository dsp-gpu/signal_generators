#pragma once

// ============================================================================
// CwGenerator — генератор Continuous Wave (комплексной синусоиды) на OpenCL
//
// ЧТО:    Генератор простейшего тонового сигнала s(t) = A·exp(j·(2π·f·t + φ))
//         на GPU и CPU. Для multi-beam режима частоты разнесены: f_i = f0 +
//         i·freq_step. Реализует общий интерфейс ISignalGenerator (CW kind).
//
// ЗАЧЕМ:  CW — опорный сигнал для тестов FFT, оконных функций, beamforming'а
//         и калибровки тракта. Без чистой синусоиды нечем верифицировать
//         spectrum-модуль (один пик в спектре = один частотный bin).
//         Multi-beam позволяет одновременно генерировать набор тонов для
//         тестов нескольких лучей в radar-pipeline.
//
// ПОЧЕМУ: - Это OpenCL-вариант (legacy nvidia-ветка) под `#if !ENABLE_ROCM`
//           неявным образом — файл не имеет guard'а, но используется только
//           когда ROCm отключён. ROCm-вариант: CwGeneratorROCm.
//         - Move-only: cl_context/queue/program уникальны на инстанс,
//           копирование = double-release GPU-ресурсов.
//         - backend не владеет (raw указатель) — DrvGPU создан выше по стеку
//           и переживает генератор.
//         - CPU-реализация GenerateToCpu — для эталона / unit-тестов без GPU.
//
// Использование:
//   signal_gen::CwGenerator gen(backend, CwParams{.f0=1e6, .amplitude=1.0});
//   auto out = gen.GenerateToGpu(system, beam_count);
//   // out — cl_mem с сигналом, caller вызывает clReleaseMemObject(out).
//
// История:
//   - Создан: 2026-02-13 (legacy OpenCL-ветка)
// ============================================================================

#include <signal_generators/i_signal_generator.hpp>
#include <core/interface/i_backend.hpp>

#include <CL/cl.h>
#include <string>
#include <cstring>
#include <utility>
#include <vector>

namespace signal_gen {

/**
 * @class CwGenerator
 * @brief OpenCL-генератор CW (комплексной синусоиды) с поддержкой multi-beam.
 *
 * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note Доступен только в OpenCL-сборке. ROCm-вариант: CwGeneratorROCm.
 * @see signal_gen::CwGeneratorROCm
 * @see signal_gen::ISignalGenerator
 */
class CwGenerator : public ISignalGenerator {
public:
    /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
    using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

    /**
     * @brief Конструктор
     * @param backend GPU backend (не владеет!)
     * @param params  Параметры CW сигнала
     */
    CwGenerator(drv_gpu_lib::IBackend* backend, const CwParams& params);
    ~CwGenerator() override;

    // Запрет копирования
    CwGenerator(const CwGenerator&) = delete;
    CwGenerator& operator=(const CwGenerator&) = delete;

    // Перемещение
    CwGenerator(CwGenerator&& other) noexcept;
    CwGenerator& operator=(CwGenerator&& other) noexcept;

    // ISignalGenerator
    void GenerateToCpu(const SystemSampling& system,
                       std::complex<float>* out, size_t out_size) override;

    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count = 1) override;

    /**
     * @brief Генерация на GPU с опциональным сбором событий профилирования
     * @param prof_events nullptr → production (zero overhead); &vec → benchmark
     *
     * Собирает события: "Kernel" (cw_kernel)
     */
    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count,
                         ProfEvents* prof_events);

    SignalKind Kind() const override { return SignalKind::CW; }

    /// Обновить параметры CW
    void SetParams(const CwParams& params) { params_ = params; }
    const CwParams& GetParams() const { return params_; }

private:
    void CompileKernel();
    void ReleaseGpuResources();

    drv_gpu_lib::IBackend* backend_ = nullptr;
    CwParams params_;

    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_program program_ = nullptr;
};

} // namespace signal_gen
