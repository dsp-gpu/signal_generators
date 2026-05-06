---
schema_version: 1
kind: use_case
id: form_signal_rocm
repo: signal_generators
title: "Формирование сигнала на GPU с возможностью сравнения с CPU"
synonyms:
  ru:
    - "Генерация сигнала на GPU"
    - "Сравнение GPU и CPU"
    - "Формирование сигнала с антеннами"
    - "Обработка сигналов в реальном времени"
    - "Генерация форматированного сигнала"
    - "Тестирование ROCm backend"
    - "Создание сигнала с фазовым сдвигом"
    - "Параллельная обработка сигналов"
  en:
    - "GPU signal generation"
    - "CPU-GPU comparison test"
    - "Antenna array signal formation"
    - "Real-time signal processing"
    - "Formatted signal generation"
    - "ROCm backend testing"
    - "Phase-shifted signal creation"
    - "Parallel signal processing"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - spectrum__filters_rocm__usecase__v1
  - spectrum__fir_basic__usecase__v1
  - spectrum__fft_processor_rocm__usecase__v1
maturity: stable
language: cpp
tags: [signal_generators, rocm, gpu_signal_generation, cpu_gpu_comparison, antenna_array, benchmarking, signal_processing, batch_processing, fft, dsp]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Формирование сигнала на GPU с возможностью сравнения с CPU

## Когда применять

Когда требуется генерировать сигналы с антеннами на GPU с последующим сравнением с CPU-референсом для тестирования производительности или корректности

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  int gpu_id = 0;

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  TestRunner runner(&backend, "FormSignal ROCm", gpu_id);

  // ── Test 1: No noise, 1 channel — GPU vs CPU reference ────────

  runner.test("no_noise", [&]() {
    FormParams p;
    p.fs = 12e6;
    p.antennas = 1;
    p.points = 4096;
    p.f0 = 1e6;
    p.amplitude = 1.0;
    p.noise_amplitude = 0.0;
    p.phase = 0.3;
    p.fdev = 0.0;
    p.norm = 1.0 / std::sqrt(2.0);

    FormSignalGeneratorROCm gen(&backend);
    gen.SetParams(p);
    auto gpu_data = gen.GenerateToCpu();

    auto cpu_ref = refs::GenerateFormSignal(
        static_cast<float>(p.fs), p.points, static_cast<float>(p.f0),
        static_cast<float>(p.amplitude), static_cast<float>(p.phase),
        static_cast<float>(p.fdev), static_cast<float>(p.norm), 0.0f);

// ... (truncated)
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [spectrum__filters_rocm__usecase__v1](./filters_rocm.md)
- См. [spectrum__fir_basic__usecase__v1](./fir_basic.md)
- См. [spectrum__fft_processor_rocm__usecase__v1](./fft_processor_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/signal_generators/tests/test_form_signal_rocm.hpp:1`
