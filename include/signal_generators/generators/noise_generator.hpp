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

    void GenerateToCpu(const SystemSampling& system,
                       std::complex<float>* out, size_t out_size) override;

    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count = 1) override;

    /**
     * @brief Генерация на GPU с опциональным сбором событий профилирования
     * @param prof_events nullptr → production (zero overhead); &vec → benchmark
     *
     * Собирает события: "Kernel" (noise_kernel / Philox+BoxMuller)
     */
    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count,
                         ProfEvents* prof_events);

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
