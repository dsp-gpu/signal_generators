---
schema_version: 1
kind: use_case
id: signal_generators_rocm_basic
repo: signal_generators
title: "Signal Generators Rocm Basic"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - signal_generators__signal_generators_benchmark_rocm__usecase__v1
  - signal_generators__form_signal_rocm__usecase__v1
  - core__rocm_backend__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Signal Generators Rocm Basic

## Когда применять

_LLM-fallback: см. описание класса._

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
- См. [core__rocm_backend__usecase__v1](./rocm_backend.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/signal_generators/tests/test_signal_generators_rocm_basic.hpp:1`
