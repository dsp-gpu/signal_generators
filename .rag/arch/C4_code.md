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
| `SignalGeneratorFactory` | **Factory** | /**
@class SignalGeneratorFactory
Создаёт генераторы по типу сигнала
/ |

## HIP-ядра (`kernels/rocm/`)

- `form_signal.hip`

## Все key_classes (FQN список)

- `signal_gen::CwGeneratorROCm` (11 методов)
- `signal_gen::CwGeneratorROCm` (11 методов)
- `signal_gen::LfmGeneratorROCm` (11 методов)
- `signal_gen::NoiseGeneratorROCm` (11 методов)
- `signal_gen::SignalGeneratorFactory` (7 методов)
- `signal_gen::DelayedFormSignalGeneratorROCm` (16 методов)
- `signal_gen::FormSignalGeneratorROCm` (25 методов)
- `signal_gen::FormSignalGeneratorROCm` (25 методов)
- `signal_gen::LfmGeneratorAnalyticalDelayROCm` (18 методов)
- `signal_gen::ScriptGeneratorROCm` (24 методов)
