#pragma once

// ============================================================================
// ISignalGenerator — интерфейс генератора сигналов (Strategy)
//
// ЧТО:    Pure-virtual интерфейс из 3 методов: GenerateToCpu (reference) /
//         GenerateToGpu (production) / Kind(). Реализуют все генераторы
//         модуля — CwGenerator, LfmGenerator, NoiseGenerator. Через
//         ISignalGenerator* SignalService и Python-биндинги работают со
//         всеми типами сигналов одинаково.
//
// ЗАЧЕМ:  Без общего интерфейса фасад / тесты / биндинги дублировали бы
//         диспетчеризацию по SignalKind. Через ISignalGenerator caller
//         делает просто `gen->GenerateToGpu(system, beam_count)` — фабрика
//         (SignalGeneratorFactory) сама выбрала конкретную стратегию.
//         Парный CPU-метод даёт reference для сверки GPU-результата.
//
// ПОЧЕМУ: - Strategy (GoF): новый сигнал = новый класс, без правок фасада
//           (OCP). Альтернатива (switch по SignalKind) ломает OCP.
//         - GenerateToCpu и GenerateToGpu обе в интерфейсе → каждый
//           генератор обязан иметь reference (LSP не нарушается, тесты
//           гарантированы).
//         - GenerateToGpu возвращает cl_mem (caller владеет, должен сам
//           clReleaseMemObject) — не RAII-обёртка, потому что результат
//           передаётся в чужой pipeline (fft_func, radar) который сам
//           решает срок жизни.
//         - Kind() для introspection в Python-биндингах и логах.
//
// Использование:
//   class MyGen : public ISignalGenerator {
//     void GenerateToCpu(const SystemSampling& sys, std::complex<float>* out,
//                        size_t out_size) override { ... }
//     cl_mem GenerateToGpu(const SystemSampling& sys, size_t beam_count) override { ... }
//     SignalKind Kind() const override { return SignalKind::CW; }
//   };
//
//   auto gen = SignalGeneratorFactory::CreateCw(backend, params);
//   cl_mem gpu_buf = gen->GenerateToGpu({1000.0, 4096}, 8);
//   // ... передать в pipeline ...
//   clReleaseMemObject(gpu_buf);
//
// История:
//   - Создан:  2026-02-13
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#include <signal_generators/params/signal_request.hpp>
#include <CL/cl.h>
#include <vector>
#include <complex>
#include <cstddef>

namespace signal_gen {

/**
 * @class ISignalGenerator
 * @brief Pure-virtual контракт всех генераторов сигналов (Strategy).
 *
 * @note Pure interface — нельзя инстанцировать. Все методы обязательны.
 * @note GenerateToGpu возвращает cl_mem с переданным владением: caller
 *       обязан clReleaseMemObject.
 * @see SignalGeneratorFactory
 * @see SignalService
 */
class ISignalGenerator {
public:
    virtual ~ISignalGenerator() = default;

    /**
     * @brief Генерация на CPU (reference implementation)
     *
     * @param system  Параметры дискретизации (fs, length)
     * @param out     Выходной буфер [length] complex<float>
     * @param out_size Размер выходного буфера (должен >= length)
     */
    virtual void GenerateToCpu(
        const SystemSampling& system,
        std::complex<float>* out,
        size_t out_size) = 0;

    /**
     * @brief Генерация на GPU (production)
     *
     * @param system      Параметры дискретизации
     * @param beam_count  Количество лучей (1 = один сигнал)
     *   @test { range=[1..50000], value=128, unit="лучей/каналов" }
     * @return cl_mem буфер [beam_count * length * sizeof(complex<float>)]
     *
     * @note Вызывающий код должен освободить cl_mem через clReleaseMemObject()!
     *   @test_check result != nullptr (cl_mem buffer with [beam_count × length × complex<float>])
     */
    virtual cl_mem GenerateToGpu(
        const SystemSampling& system,
        size_t beam_count = 1) = 0;

    /**
     * @brief Возвращает тип сигнала (для introspection в Python-биндингах и логах).
     *
     * @return Один из SignalKind::CW/LFM/NOISE/FORM_SIGNAL.
     *   @test_check result == SignalKind::{CW|LFM|NOISE|FORM_SIGNAL}
     */
    virtual SignalKind Kind() const = 0;
};

} // namespace signal_gen
