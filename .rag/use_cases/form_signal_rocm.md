---
schema_version: 1
kind: use_case
id: form_signal_rocm
repo: signal_generators
title: "Form Signal Rocm"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - spectrum__filters_rocm__usecase__v1
  - core__rocm_backend__usecase__v1
  - stats__statistics_rocm__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Form Signal Rocm

## Когда применять

_LLM-fallback: см. описание класса._

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
- См. [core__rocm_backend__usecase__v1](./rocm_backend.md)
- См. [stats__statistics_rocm__usecase__v1](./statistics_rocm.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/signal_generators/tests/test_form_signal_rocm.hpp:1`
