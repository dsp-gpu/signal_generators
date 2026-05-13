---
schema_version: 1
repo: signal_generators
arch_level: c4
tags:
  - "#level:c4"
  - "#repo:signal_generators"
  - "#layer:compute"
  - "#pattern:Factory:SignalGeneratorFactory"
description: "C4 Code — реальные классы с паттернами GoF/SOLID для репо signal_generators."
---

# C4 Code — `signal_generators`

## Классы с паттернами проектирования

| Класс | Паттерн | Brief |
|-------|---------|-------|
| `SignalGeneratorFactory` | **Factory** | генераторы по типу сигнала |

## HIP-ядра (`kernels/rocm/`)

- `form_signal.hip`

## Все key_classes (FQN список)

- `dsp::signal_generators::CwGeneratorROCm` (11 методов)
- `dsp::signal_generators::NoiseGeneratorROCm` (11 методов)
- `dsp::signal_generators::LfmGeneratorROCm` (11 методов)
- `dsp::signal_generators::SignalGeneratorFactory` (7 методов)
- `dsp::signal_generators::DelayedFormSignalGeneratorROCm` (16 методов)
- `dsp::signal_generators::FormSignalGeneratorROCm` (25 методов)
- `dsp::signal_generators::LfmGeneratorAnalyticalDelayROCm` (18 методов)
- `dsp::signal_generators::ScriptGeneratorROCm` (24 методов)
- `dsp::signal_generators::FormScriptGeneratorROCm` (14 методов)
- `dsp::signal_generators::LfmConjugateGeneratorROCm` (13 методов)
