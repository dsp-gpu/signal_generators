---
schema_version: 1
kind: use_case
id: signal_generators_benchmark_rocm
repo: signal_generators
title: "Как выполнить бенчмаркинг генератора сигналов на GPU ROCm"
synonyms:
  ru:
    - "бенчмаркинг генератора сигналов на ROCm"
    - "тестирование производительности генератора сигналов"
    - "оценка скорости генерации сигналов на GPU"
    - "benchmark антенных массивов на ROCm"
    - "тестирование шумовых сигналов на GPU"
    - "измерение производительности ROCm для сигналов"
    - "оценка времени генерации сигналов"
    - "тестирование ROCm для радиолокационных сигналов"
  en:
    - "benchmark signal generator ROCm"
    - "performance testing signal generator"
    - "gpu signal generation benchmark"
    - "antenna array benchmark ROCm"
    - "noise signal testing on GPU"
    - "rocml signal processing benchmark"
    - "signal generation speed measurement"
    - "gpu benchmark for radar signals"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - signal_generators__form_signal_rocm__usecase__v1
  - spectrum__lch_farrow_rocm__usecase__v1
  - spectrum__filters_benchmark_rocm__usecase__v1
maturity: stable
language: cpp
tags: [signal_generators, rocml, benchmark, signal_generator, gpu, performance_testing, antenna_array, noise_processing, dsp]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Как выполнить бенчмаркинг генератора сигналов на GPU ROCm

## Когда применять

Когда нужно протестировать производительность генератора сигналов на ROCm с поддержкой антенных массивов и шумов

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  std::cout << "\n"
            << "============================================================\n"
            << "  FormSignalGeneratorROCm Benchmark (GpuBenchmarkBase)\n"
            << "============================================================\n";

  // Проверить AMD GPU
  if (drv_gpu_lib::ROCmCore::GetAvailableDeviceCount() == 0) {
    std::cout << "  [SKIP] No AMD GPU available\n";
    return 0;
  }

  try {
    // ── ROCm backend init ─────────────────────────────────────────────────
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    // ── Параметры генератора ───────────────────────────────────────────────
    dsp::signal_generators::FormParams params;
    params.fs              = 12e6;
    params.antennas        = 8;
    params.points          = 4096;
    params.f0              = 1e6;
    params.amplitude       = 1.0;
    params.noise_amplitude = 0.0;

    // ── Создать генератор ──────────────────────────────────────────────────
    dsp::signal_generators::FormSignalGeneratorROCm gen(backend.get());
    gen.SetParams(params);

    // ── Benchmark ─────────────────────────────────────────────────────────
// ... (truncated)
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [signal_generators__form_signal_rocm__usecase__v1](./form_signal_rocm.md)
- См. [spectrum__lch_farrow_rocm__usecase__v1](./lch_farrow_rocm.md)
- См. [spectrum__filters_benchmark_rocm__usecase__v1](./filters_benchmark_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/signal_generators/tests/test_signal_generators_benchmark_rocm.hpp:1`
