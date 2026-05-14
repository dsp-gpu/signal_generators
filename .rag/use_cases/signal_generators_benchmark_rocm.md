---
schema_version: 1
kind: use_case
id: signal_generators_benchmark_rocm
repo: signal_generators
title: "Signal Generators Benchmark Rocm"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - spectrum__filters_benchmark_rocm__usecase__v1
  - signal_generators__form_signal_rocm__usecase__v1
  - stats__statistics_rocm__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Signal Generators Benchmark Rocm

## Когда применять

_LLM-fallback: см. описание класса._

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

- См. [spectrum__filters_benchmark_rocm__usecase__v1](./filters_benchmark_rocm.md)
- См. [signal_generators__form_signal_rocm__usecase__v1](./form_signal_rocm.md)
- См. [stats__statistics_rocm__usecase__v1](./statistics_rocm.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/signal_generators/tests/test_signal_generators_benchmark_rocm.hpp:1`
