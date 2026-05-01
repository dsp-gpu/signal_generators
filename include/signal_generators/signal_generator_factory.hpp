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
    /// Создать CW генератор
    static std::unique_ptr<ISignalGenerator> CreateCw(
        drv_gpu_lib::IBackend* backend, const CwParams& params);

    /// Создать LFM генератор
    static std::unique_ptr<ISignalGenerator> CreateLfm(
        drv_gpu_lib::IBackend* backend, const LfmParams& params);

    /// Создать Noise генератор
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

    /// Создать FormScriptGenerator (DSL + on-disk cache)
    static std::unique_ptr<FormScriptGenerator> CreateFormScript(
        drv_gpu_lib::IBackend* backend, const FormParams& params);

    /// Создать генератор по SignalRequest
    static std::unique_ptr<ISignalGenerator> Create(
        drv_gpu_lib::IBackend* backend, const SignalRequest& request);
};

} // namespace signal_gen
