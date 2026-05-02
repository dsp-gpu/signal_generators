#pragma once

// ============================================================================
// all_test.hpp — точка входа в тесты модуля signal_generators
//
// ЧТО:    Агрегирует все тест-файлы (ROCm): basic-генераторы, FormSignal,
//         бенчмарки. main.cpp вызывает только этот файл.
// ЗАЧЕМ:  Единый список — включить/отключить тест без правки main.cpp.
// ПОЧЕМУ: Следует схеме {repo}/tests/all_test.hpp (правило 15-cpp-testing).
//
// История: Создан: 2026-02-17
// ============================================================================

/**
 * @file all_test.hpp
 * @brief Точка входа в тесты модуля signal_generators.
 * @note Не публичный API. Вызывается из main.cpp через signal_generators_all_test::run().
 */

#if ENABLE_ROCM
#include "test_signal_generators_rocm_basic.hpp"
#include "test_form_signal_rocm.hpp"
#include "test_signal_generators_benchmark_rocm.hpp"
#endif

namespace signal_generators_all_test {

inline void run() {
#if ENABLE_ROCM
    // CW, LFM, Noise, LfmConjugate: GPU vs CPU
    test_signal_generators_rocm_basic::run();

    // FormSignalGeneratorROCm: getX on HIP
    test_form_signal_rocm::run();

    // ROCm Benchmarks (GpuBenchmarkBase)
    //   test_signal_generators_benchmark_rocm::run();
#endif
}

}  // namespace signal_generators_all_test
