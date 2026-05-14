<!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->

# signal_generators — Repository Card

_Источник: `signal_generators/CLAUDE.md`_

# 🤖 CLAUDE — `signal_generators`

> Генераторы сигналов: CW, LFM, Noise, Script, FormSignal.
> Зависит от: `core` + `spectrum`. Глобальные правила → `../CLAUDE.md` + `.claude/rules/*.md`.

## 🎯 Что здесь

| Класс | Что делает |
|-------|-----------|
| `CWGenerator` | Continuous Wave (одна частота) |
| `LFMGenerator` | Linear Frequency Modulation (chirp) |
| `NoiseGenerator` | White noise (rocRAND или Philox inline) |
| `ScriptGenerator` | Импорт сигнала из JSON-скрипта |
| `FormSignalGenerator` | Композит: CW + LFM + Noise + ScriptSegments |

## 📁 Структура

```
signal_generators/
├── include/dsp/signal_generators/
│   ├── signal_generator.hpp        # базовый (abstract)
│   ├── form_signal_generator.hpp   # facade/composite
│   ├── operations/                 # CWOp, LFMOp, NoiseOp, ScriptOp
│   └── strategies/                 # NoiseStrategy (Philox vs rocRAND)
├── src/
├── kernels/rocm/                   # cw.hip, lfm.hip, noise_philox.hip
├── tests/
└── python/dsp_signal_generators_module.cpp
```

## ⚠️ Специфика

- **Phase accumulator**: для длинных сигналов накопление фазы в `double` → потом `float`, иначе дрейф.
- **LFM chirp**: `phase = 2π(f0·t + 0.5·k·t²)`. Для высоких k/длинных T — двойная точность фазы.
- **Noise**: inline Philox-4x32-10 даёт воспроизводимость при фиксированном seed — предпочтительно для тестов.
- **FormSignal** — composite pattern, сегменты определяются в JSON / struct.

## 🚫 Запреты

- Не использовать `rand()` / `std::rand()` на хосте — воспроизводимость ломается.
- Не генерировать сигнал на CPU и копировать на GPU — только нативный HIP kernel.
- Не встраивать LFM Dechirp сюда — это модуль `heterodyne`.

<!-- BEGIN: RAG_CLAUDE_C4 (auto) -->
## 🏗️ Архитектура (C4 — компактно)

- **C1 System Context:** репо `signal_generators` (layer=compute). Полный C4 → `MemoryBank/.architecture/DSP-GPU_Design_C4_Full.md` §`signal_generators`
- **C2 Container:** namespace из top key_classes (см. `.rag/_RAG.md`)
- **C3 Component:** `key_classes` в `.rag/_RAG.md` (top по test_params)
- **C4 Code:** SignalGeneratorFactory · CwGeneratorROCm · LfmGeneratorROCm · NoiseGeneratorROCm

## 🏷️ RAG теги


`#layer:compute` `#repo:signal_generators` `#namespace:signal_gen` `#pattern:Factory:SignalGeneratorFactory` `#pattern:Facade:FormSignalGenerator` `#pattern:Facade:ScriptGenerator` `#pattern:Strategy:ISignalGenerator`

## 🔗 Правила (path-scoped автоматически)

- `09-rocm-only.md`
- `05-architecture-ref03.md` — composite через FormSignal
- `14-cpp-style.md` + `15-cpp-testing.md`
- `11-python-bindings.md`
