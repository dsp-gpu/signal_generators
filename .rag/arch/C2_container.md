---
schema_version: 1
repo: signal_generators
arch_level: c2
tags:
  - "#level:c2"
  - "#repo:signal_generators"
  - "#layer:compute"
  - "#namespace:dsp_signal_generators"
  - "#namespace:test_signal_generators_rocm"
description: "C2 Container — namespace tree и зависимости репо signal_generators."
---

# C2 Container — `signal_generators` (layer=compute)

## Namespaces (top по числу классов)

- `signal_gen`
- `test_signal_generators_rocm`

## Public modules (`include/signal_generators/`)

- `generators/`
- `kernels/`
- `params/`

## Зависимости (depends_on)

`core`

## Используется (used_by)

`DSP`

## Top key_classes

| Class | Namespace | Methods | TestParams |
|-------|-----------|--------:|-----------:|
| `CwGeneratorROCm` | `signal_gen` | 11 | 19 |
| `NoiseGeneratorROCm` | `signal_gen` | 11 | 16 |
| `LfmGeneratorROCm` | `signal_gen` | 11 | 16 |
| `SignalGeneratorFactory` | `signal_gen` | 7 | 15 |
| `DelayedFormSignalGeneratorROCm` | `signal_gen` | 16 | 7 |
