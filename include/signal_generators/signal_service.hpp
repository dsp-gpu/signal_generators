#pragma once

#if !ENABLE_ROCM  // OpenCL-only facade: ROCm modules use generators directly

// ============================================================================
// SignalService — Facade модуля signal_generators (OpenCL-only)
//
// ЧТО:    Высокоуровневый фасад с перегрузками GenerateCpu / GenerateGpu для
//         всех типов сигналов (CwParams / LfmParams / NoiseParams / FormParams)
//         и отдельными API GenerateFormGpu / GenerateDelayedFormGpu для
//         мультиканальных Form-сценариев (Farrow 48×5 fractional delay).
//         Кеширует генераторы в std::optional — перекомпиляция HIP/CL ядра
//         только при смене параметров.
//
// ЗАЧЕМ:  Пользователь (тесты, Python-биндинги, fft_func / radar pipeline)
//         не должен:
//         - вручную создавать генератор через factory;
//         - помнить про cache invalidation при смене params;
//         - выбирать между ISignalGenerator API (CW/LFM/Noise) и
//           специализированным API Form-генераторов;
//         - управлять lifecycle backend'а в каждом вызове.
//         Service всё это инкапсулирует и даёт перегрузки по типу params.
//
// ПОЧЕМУ: - Facade (GoF) + GRASP Controller: координирует factory + кеш +
//           backend, не делает работу сам. Каждая перегрузка — 5-15 строк.
//         - std::optional<Generator> вместо unique_ptr → объект живёт в
//           самом Service'е (no heap fragmentation), reset() = смена params.
//         - operator!= в *Params → дешёвое сравнение перед reset() (вместо
//           всегда пересоздавать).
//         - Перегрузка по типу params (не enum dispatch) → compile-time
//           type safety, нельзя передать NoiseParams в CW-метод.
//         - Form/Delayed Form → отдельные методы, возвращают InputData<cl_mem>
//           для прямой стыковки с fft_func без копирования.
//         - #if !ENABLE_ROCM на весь файл: SignalService — OpenCL-фасад;
//           ROCm-модули используют генераторы напрямую (другая модель
//           lifecycle). Не пытаться унифицировать — это разные архитектуры.
//
// Использование:
//   signal_gen::SignalService service(backend);
//   signal_gen::CwParams cw{.f0 = 100.0, .freq_step = 10.0};
//   auto cpu_data = service.GenerateCpu(cw, {1000.0, 4096});
//   cl_mem gpu_data = service.GenerateGpu(cw, {1000.0, 4096}, 8);
//   // ... использовать ...
//   clReleaseMemObject(gpu_data);  // владение — у caller'а
//
// История:
//   - Создан:  2026-02-13
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#include <signal_generators/signal_generator_factory.hpp>
#include <signal_generators/params/signal_request.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <signal_generators/generators/cw_generator.hpp>
#include <signal_generators/generators/lfm_generator.hpp>
#include <signal_generators/generators/noise_generator.hpp>
#include <signal_generators/generators/form_signal_generator.hpp>
#include <signal_generators/generators/delayed_form_signal_generator.hpp>

#include <CL/cl.h>
#include <memory>
#include <optional>
#include <vector>
#include <complex>

namespace signal_gen {

/**
 * @class SignalService
 * @brief Facade для генерации сигналов на CPU/GPU (OpenCL backend).
 *
 * @note Кеширует генераторы (std::optional) — перекомпиляция ядра только
 *       при смене params (через operator!=).
 * @note GenerateGpu / GenerateFormGpu возвращают cl_mem с переданным
 *       владением: caller обязан clReleaseMemObject.
 * @note Доступен только при !ENABLE_ROCM. ROCm-модули используют
 *       генераторы напрямую через SignalGeneratorFactory::Create*ROCm.
 * @see SignalGeneratorFactory
 * @see ISignalGenerator
 *
 * @code
 * signal_gen::SignalService service(backend);
 *
 * signal_gen::CwParams cw;
 * cw.f0 = 100.0;
 * cw.freq_step = 10.0;
 *
 * // CPU
 * auto cpu_data = service.GenerateCpu(cw, {1000.0, 4096});
 *
 * // GPU
 * cl_mem gpu_data = service.GenerateGpu(cw, {1000.0, 4096}, 8);
 * // ... использовать gpu_data ...
 * clReleaseMemObject(gpu_data);
 * @endcode
 */
class SignalService {
public:
    explicit SignalService(drv_gpu_lib::IBackend* backend)
        : backend_(backend) {}

    // ═══════════════════════════════════════════════════════════════════
    // CPU generation
    // ═══════════════════════════════════════════════════════════════════

    /// Генерация CW на CPU (1 луч)
    std::vector<std::complex<float>> GenerateCpu(
        const CwParams& params, const SystemSampling& system);

    /// Генерация LFM на CPU (1 луч)
    std::vector<std::complex<float>> GenerateCpu(
        const LfmParams& params, const SystemSampling& system);

    /// Генерация Noise на CPU (1 луч)
    std::vector<std::complex<float>> GenerateCpu(
        const NoiseParams& params, const SystemSampling& system);

    // ═══════════════════════════════════════════════════════════════════
    // GPU generation
    // ═══════════════════════════════════════════════════════════════════

    /// Генерация CW на GPU (N лучей)
    cl_mem GenerateGpu(const CwParams& params, const SystemSampling& system,
                       size_t beam_count = 1);

    /// Генерация LFM на GPU (N лучей)
    cl_mem GenerateGpu(const LfmParams& params, const SystemSampling& system,
                       size_t beam_count = 1);

    /// Генерация Noise на GPU (N лучей)
    cl_mem GenerateGpu(const NoiseParams& params, const SystemSampling& system,
                       size_t beam_count = 1);

    // ═══════════════════════════════════════════════════════════════════
    // FormSignal generation (standalone API)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @brief Генерация FormSignal на GPU (мультиканальная)
     * @param params FormParams (fs, antennas, points уже внутри)
     * @return InputData<cl_mem> — совместимо с fft_func (data, antenna_count, n_point, gpu_memory_bytes)
     * @note Вызывающий код должен освободить result.data через clReleaseMemObject()
     */
    drv_gpu_lib::InputData<cl_mem> GenerateFormGpu(const FormParams& params);

    /**
     * @brief Генерация FormSignal на CPU (по каналам)
     * @param params FormParams
     * @return vector[antenna_id][sample_id] complex<float>
     */
    std::vector<std::vector<std::complex<float>>> GenerateFormCpu(
        const FormParams& params);

    // ═══════════════════════════════════════════════════════════════════
    // DelayedFormSignal generation (Farrow 48×5)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @brief Генерация сигнала с дробной задержкой (Farrow) на GPU
     * @param params FormParams (fs, antennas, points, noise_amplitude)
     * @param delay_us Задержки per-antenna в микросекундах
     * @return InputData<cl_mem>
     * @note Вызывающий код должен освободить result.data через clReleaseMemObject()
     */
    drv_gpu_lib::InputData<cl_mem> GenerateDelayedFormGpu(
        const FormParams& params, const std::vector<float>& delay_us);

    /**
     * @brief Генерация сигнала с дробной задержкой (Farrow) на CPU
     * @param params FormParams
     * @param delay_us Задержки per-antenna в микросекундах
     * @return vector[antenna_id][sample_id] complex<float>
     */
    std::vector<std::vector<std::complex<float>>> GenerateDelayedFormCpu(
        const FormParams& params, const std::vector<float>& delay_us);

private:
    drv_gpu_lib::IBackend* backend_;

    // Кешированные генераторы: перекомпиляция ядра только при смене params
    std::optional<CwGenerator>                  cw_gen_;
    CwParams                                    cw_params_{};
    std::optional<LfmGenerator>                 lfm_gen_;
    LfmParams                                   lfm_params_{};
    std::optional<NoiseGenerator>               noise_gen_;
    NoiseParams                                 noise_params_{};
    std::optional<FormSignalGenerator>          form_gen_;
    std::optional<DelayedFormSignalGenerator>   delayed_gen_;

    CwGenerator&    GetCw(const CwParams& p);
    LfmGenerator&   GetLfm(const LfmParams& p);
    NoiseGenerator& GetNoise(const NoiseParams& p);
};

} // namespace signal_gen

#endif  // !ENABLE_ROCM
