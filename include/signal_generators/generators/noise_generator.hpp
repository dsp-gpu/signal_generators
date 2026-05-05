#pragma once

// ============================================================================
// NoiseGenerator — генератор гауссовского комплексного шума на OpenCL
//
// ЧТО:    Белый комплексный шум s(t) = (n_re + j·n_im), где n_re, n_im ~
//         N(0, σ²). GPU: Philox-2x32-10 PRNG + Box-Muller transform для
//         нормального распределения. CPU: std::mt19937 + std::normal_dist
//         (эталон). Реализует ISignalGenerator (kind=NOISE).
//
// ЗАЧЕМ:  Шум — обязательный компонент тестирования radar/DSP-пайплайнов:
//         оценка SNR, проверка устойчивости детекторов, симуляция
//         реалистичных условий приёма. Воспроизводимость по seed критична
//         для unit-тестов: одинаковый seed → один и тот же шум на всех
//         запусках (без неё тесты «плавают»).
//
// ПОЧЕМУ: - OpenCL-вариант (legacy nvidia-ветка). ROCm-вариант:
//           NoiseGeneratorROCm. На main-ветке main = ROCm.
//         - Philox-2x32 (counter-based PRNG) — параллелится без коллизий
//           (каждая нить берёт уникальный counter), в отличие от LCG.
//         - Box-Muller — стандартное преобразование U(0,1) → N(0,1).
//         - Move-only: cl_program/queue/context/kernel уникальны на инстанс.
//         - backend не владеет — caller гарантирует переживание объекта.
//
// Использование:
//   signal_gen::NoiseGenerator gen(backend,
//       NoiseParams{.amplitude = 1.0f, .seed = 42});
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
 * @class NoiseGenerator
 * @brief OpenCL-генератор Gaussian-шума (Philox-2x32 + Box-Muller).
 *
 * @note Move-only: GPU-ресурсы уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note Воспроизводимость через seed обязательна для unit-тестов.
 * @note OpenCL-вариант. ROCm-аналог: NoiseGeneratorROCm.
 * @see signal_gen::NoiseGeneratorROCm
 * @see signal_gen::ISignalGenerator
 */
class NoiseGenerator : public ISignalGenerator {
public:
    /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
    using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

    NoiseGenerator(drv_gpu_lib::IBackend* backend, const NoiseParams& params);
    ~NoiseGenerator() override;

    NoiseGenerator(const NoiseGenerator&) = delete;
    NoiseGenerator& operator=(const NoiseGenerator&) = delete;
    NoiseGenerator(NoiseGenerator&& other) noexcept;
    NoiseGenerator& operator=(NoiseGenerator&& other) noexcept;

    /**
     * @brief CPU reference генерация шума (white или Gaussian по NoiseParams::type).
     *
     * @param system Параметры дискретизации (fs, length).
     * @param out Выходной буфер [out_size] complex<float>.
     * @param out_size Размер буфера (должен быть >= system.length).
     */
    void GenerateToCpu(const SystemSampling& system,
                       std::complex<float>* out, size_t out_size) override;

    /**
     * @brief GPU production: Philox+BoxMuller kernel, multi-beam.
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
     * Собирает события: "Kernel" (noise_kernel / Philox+BoxMuller)
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
     * @return Всегда `SignalKind::NOISE` для этого класса.
     *   @test_check result == SignalKind::NOISE
     */
    SignalKind Kind() const override { return SignalKind::NOISE; }

    void SetParams(const NoiseParams& params) { params_ = params; }
    const NoiseParams& GetParams() const { return params_; }

private:
    void CompileKernel();
    void ReleaseGpuResources();

    drv_gpu_lib::IBackend* backend_ = nullptr;
    NoiseParams params_;

    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
};

} // namespace signal_gen
