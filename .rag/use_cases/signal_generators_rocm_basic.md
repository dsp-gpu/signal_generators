---
schema_version: 1
kind: use_case
id: signal_generators_rocm_basic
repo: signal_generators
title: "Как генерировать CW сигналы на GPU"
synonyms:
  ru:
    - "генерация CW сигналов на GPU"
    - "сравнение GPU и CPU для CW"
    - "многолучевые сигналы на GPU"
    - "генерация сигналов батчем"
    - "обработка сигналов ROCm"
    - "сигналы с частотным шагом"
    - "тестирование GPU для CW"
    - "вычисления на GPU для антенн"
  en:
    - "generate cw signals on gpu"
    - "compare gpu and cpu for cw"
    - "multi-beam signals on gpu"
    - "signal generation batch"
    - "signal processing rocm"
    - "frequency stepped signals"
    - "gpu testing for cw"
    - "antenna array signal generation"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - signal_generators__signal_generators_benchmark_rocm__usecase__v1
  - signal_generators__form_signal_rocm__usecase__v1
  - spectrum__filters_rocm__usecase__v1
maturity: stable
language: cpp
tags: [signal_generators, rocm, fft, batch, antenna, signal_generation, cw, gpu, benchmarking, performance]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Как генерировать CW сигналы на GPU

## Когда применять

Для тестирования производительности GPU по сравнению с CPU или генерации многолучевых сигналов с частотным шагом

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  int gpu_id = 0;

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  TestRunner runner(&backend, "SigGen ROCm", gpu_id);

  // ── Test 1: CW GPU vs CPU ─────────────────────────────────────

  runner.test("cw_gpu_vs_cpu", [&]() {
    CwParams cw;
    cw.f0 = 250.0;
    cw.amplitude = 1.5;
    cw.phase = 0.3;
    SystemSampling sys{4000.0, 4096};

    CwGeneratorROCm gen(&backend);
    auto cpu_data = gen.GenerateToCpu(sys, cw, 1);

    auto gpu_result = gen.GenerateToGpu(sys, cw, 1);
    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, sys.length);

    return AbsError(gpu_data.data(), cpu_data.data(), sys.length,
                    tolerance::kComplex32, "cw_max_err");
  });

  // ── Test 2: CW multi-beam (8 beams, freq_step) ────────────────

  runner.test("cw_multi_beam", [&]() -> TestResult {
// ... (truncated)
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [signal_generators__signal_generators_benchmark_rocm__usecase__v1](./signal_generators_benchmark_rocm.md)
- См. [signal_generators__form_signal_rocm__usecase__v1](./form_signal_rocm.md)
- См. [spectrum__filters_rocm__usecase__v1](./filters_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/signal_generators/tests/test_signal_generators_rocm_basic.hpp:1`
