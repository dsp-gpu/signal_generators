#pragma once

// ============================================================================
// SignalGeneratorFactory — Factory Method для генераторов сигналов
//
// ЧТО:    Статическая фабрика, создающая конкретные генераторы:
//         - CreateCw / CreateLfm / CreateNoise — возвращают
//           unique_ptr<ISignalGenerator> по соответствующему *Params;
//         - CreateForm[ROCm] / CreateFormScript — возвращают конкретные
//           типы (FormSignalGenerator[ROCm], FormScriptGenerator), т.к. их
//           API шире ISignalGenerator (мультиканальный, DSL, on-disk cache);
//         - Create(SignalRequest) — диспетчер по SignalKind через variant.
//
// ЗАЧЕМ:  Caller (SignalService, Python-биндинги, тесты) не должен знать
//         про конкретные классы и их платформенно-зависимые варианты
//         (FormSignalGenerator OpenCL vs FormSignalGeneratorROCm). Factory
//         скрывает выбор реализации через #if ENABLE_ROCM. Без неё каждый
//         caller плодил бы свой #if — нарушение DRY и риск рассинхрона.
//
// ПОЧЕМУ: - Factory Method (GoF) + GRASP Creator: factory знает контекст
//           (какой backend доступен, какой ctor вызвать).
//         - Static-only (без состояния) → нет нужды в инстансе фабрики.
//         - unique_ptr<ISignalGenerator> для CW/LFM/Noise → caller владеет,
//           RAII освобождает; интерфейс достаточен для всех 3 типов.
//         - Для Form-генераторов возвращается КОНКРЕТНЫЙ тип: их API
//           (мультиканал, antenna_count, FormParams) не вписывается в
//           интерфейс single-beam ISignalGenerator → ISP (узкие интерфейсы).
//         - #if ENABLE_ROCM ветвление — единственный способ выбрать backend
//           без runtime-стоимости (платформа фиксирована при компиляции).
//
// Использование:
//   // CW generator через интерфейс:
//   auto cw = SignalGeneratorFactory::CreateCw(backend, CwParams{.f0 = 1e6});
//   cl_mem buf = cw->GenerateToGpu({1000.0, 4096}, 8);
//
//   // Form generator (ROCm) через конкретный тип:
//   auto form = SignalGeneratorFactory::CreateFormROCm(
//       backend, FormParams::ParseFromString("f0=1e6,a=1.0,an=0.1"));
//   form->GenerateBatch(...);
//
//   // Универсально через SignalRequest:
//   SignalRequest req{SignalKind::LFM, {1e6, 4096}, LfmParams{}};
//   auto gen = SignalGeneratorFactory::Create(backend, req);
//
// История:
//   - Создан:  2026-02-13
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#include <signal_generators/i_signal_generator.hpp>
#include <signal_generators/params/signal_request.hpp>
#if !ENABLE_ROCM
#include <signal_generators/generators/form_signal_generator.hpp>
#endif
#if ENABLE_ROCM
#include <signal_generators/generators/form_signal_generator_rocm.hpp>
#endif
#include <signal_generators/generators/form_script_generator.hpp>
#include <core/interface/i_backend.hpp>
#include <core/common/backend_type.hpp>

#include <memory>

namespace signal_gen {

/**
 * @class SignalGeneratorFactory
 * @brief Фабрика генераторов сигналов (CW / LFM / Noise / Form / FormScript).
 *
 * @note Static-only: инстансы не создаются.
 * @note Для CW/LFM/Noise возвращает unique_ptr<ISignalGenerator>; для Form-
 *       вариантов — конкретные типы (их API шире интерфейса).
 * @see ISignalGenerator
 * @see SignalRequest
 */
class SignalGeneratorFactory {
public:
    /**
     * @brief Создаёт CW (continuous wave) генератор. Backend-вариант выбран compile-time через #if ENABLE_ROCM.
     *
     * @param backend Указатель на IBackend (non-owning, обязан быть валидным).
     *   @test { values=["valid_backend"] }
     * @param params Параметры CW (f0, phase, amplitude, complex_iq, freq_step).
     *   @test_ref CwParams
     *
     * @return unique_ptr<ISignalGenerator> с конкретной CW-реализацией; caller владеет.
     *   @test_check result != nullptr && result->Kind() == SignalKind::CW
     */
    static std::unique_ptr<ISignalGenerator> CreateCw(
        drv_gpu_lib::IBackend* backend, const CwParams& params);

    /**
     * @brief Создаёт LFM (linear frequency modulation, chirp) генератор.
     *
     * @param backend Указатель на IBackend (non-owning, обязан быть валидным).
     *   @test { values=["valid_backend"] }
     * @param params Параметры LFM (f_start, f_end, amplitude, complex_iq).
     *   @test_ref LfmParams
     *
     * @return unique_ptr<ISignalGenerator> с конкретной LFM-реализацией; caller владеет.
     *   @test_check result != nullptr && result->Kind() == SignalKind::LFM
     */
    static std::unique_ptr<ISignalGenerator> CreateLfm(
        drv_gpu_lib::IBackend* backend, const LfmParams& params);

    /**
     * @brief Создаёт Noise-генератор (white или Gaussian, выбор по NoiseParams::type).
     *
     * @param backend Указатель на IBackend (non-owning, обязан быть валидным).
     *   @test { values=["valid_backend"] }
     * @param params Параметры шума (type, power, seed).
     *   @test_ref NoiseParams
     *
     * @return unique_ptr<ISignalGenerator> с конкретной Noise-реализацией; caller владеет.
     *   @test_check result != nullptr && result->Kind() == SignalKind::NOISE
     */
    static std::unique_ptr<ISignalGenerator> CreateNoise(
        drv_gpu_lib::IBackend* backend, const NoiseParams& params);

#if !ENABLE_ROCM
    /// Создать FormSignalGenerator (OpenCL, standalone, не ISignalGenerator)
    static std::unique_ptr<FormSignalGenerator> CreateForm(
        drv_gpu_lib::IBackend* backend, const FormParams& params);
#endif

#if ENABLE_ROCM
    /// Создать FormSignalGeneratorROCm (ROCm, standalone)
    static std::unique_ptr<FormSignalGeneratorROCm> CreateFormROCm(
        drv_gpu_lib::IBackend* backend, const FormParams& params);
#endif

    /**
     * @brief Создаёт FormScriptGenerator — мультиканальный генератор по DSL с on-disk кэшем.
     *
     * @param backend Указатель на IBackend (non-owning, обязан быть валидным).
     *   @test { values=["valid_backend"] }
     * @param params Параметры формы сигнала (см. FormParams::ParseFromString).
     *   @test_ref FormParams
     *
     * @return unique_ptr<FormScriptGenerator>; caller владеет.
     *   @test_check result != nullptr
     */
    static std::unique_ptr<FormScriptGenerator> CreateFormScript(
        drv_gpu_lib::IBackend* backend, const FormParams& params);

    /**
     * @brief Диспетчер по SignalRequest::kind: создаёт CW/LFM/Noise генератор через variant params.
     *
     * @param backend Указатель на IBackend (non-owning, обязан быть валидным).
     *   @test { values=["valid_backend"] }
     * @param request SignalRequest с kind + system + variant<CwParams, LfmParams, NoiseParams, FormParams>.
     *
     * @return unique_ptr<ISignalGenerator> по выбранному kind; caller владеет.
     *   @test_check result != nullptr && result->Kind() == request.kind
     */
    static std::unique_ptr<ISignalGenerator> Create(
        drv_gpu_lib::IBackend* backend, const SignalRequest& request);
};

} // namespace signal_gen
