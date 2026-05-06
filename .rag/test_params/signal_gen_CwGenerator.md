---
schema_version: 1
repo: signal_generators
class_fqn: signal_gen::CwGenerator
file: E:/DSP-GPU/signal_generators/include/signal_generators/generators/cw_generator.hpp
line: 30
brief: "Генерирует сигналы непрерывной волны (CW) для CPU и GPU. Создает cl_mem для OpenCL-ускорения."
methods_total: 4
methods_with_doxygen: 4
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['Генератор непрерывных волн', 'CW Signal Generator', 'Сигнал непрерывной волны', 'Одночастотный генератор']
synonyms_en: ['Continuous Wave Generator', 'CW Signal Generator', 'Single Frequency Generator', 'Monochromatic Signal Generator']
tags: ['GPU', 'OpenCL', 'Signal Generation', 'Radar', 'CW']
---

# `signal_gen::CwGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class CwGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__cw_generator__class_overview__v1 -->

**ЧТО**: Генерирует сигналы непрерывной волны (CW) для CPU и GPU. Создает cl_mem для OpenCL-ускорения.

**ЗАЧЕМ**: Решает задачу генерации одночастотных сигналов для радарных систем с высокой производительностью.

**КАК**: Использует OpenCL для GPU-ускорения, поддерживает многолучевые сигналы. Методы переопределяют интерфейс ISignalGenerator.

**Пример**:
```cpp
#include "signal_generators/generators/cw_generator.hpp"
using namespace signal_gen;
CwGenerator gen;
cl_mem gpu_buffer = gen.GenerateToGpu(system, 8);
std::complex<float>* cpu_data = new std::complex<float>[1024];
gen.GenerateToCpu(system, cpu_data, 1024);
```

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__architecture__section_002__v1` (section): ``` ┌─────────────────────────────────────────────────────────────┐ │                     Signal Generators                        │ ├─────────────────────────────────────────────────────────────┤ │  …
- `signal_generators__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->  # signal_generators — Repository Card  _Источник: `signal_generators/CLAUDE.md`_  # 🤖 CLAUDE — `signal_generators` …
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__gpu__section_002__v1` (section): ### Какой класс выбрать  | Класс | Для чего | Выход | Задержка | |-------|----------|-------|----------| | `FormSignalGenerator` | CW/Chirp + шум, N антенн | `InputData<cl_mem>` | FIXED/LINEAR/RANDOM …

## Public-методы (4)

## Method 1: `GenerateToCpu`

**Сигнатура** (`cw_generator.hpp:84`):
```cpp
void GenerateToCpu(const SystemSampling& system, std::complex<float>* out, size_t out_size) override
```

**Параметры**:
- `system` — `const SystemSampling&`
- `out` — `std::complex<float>*` *(pointer)*
- `out_size` — `size_t`

**Doxygen-источник**:
```cpp
/**
     * @brief CPU reference генерация CW (continuous wave) сигнала.
     *
     * @param system Параметры дискретизации (fs, length).
     * @param out Выходной буфер [out_size] complex<float>.
     * @param out_size Размер буфера (должен быть >= system.length).
     */
```

## Method 2: `GenerateToGpu`

**Сигнатура** (`cw_generator.hpp:97`):
```cpp
cl_mem GenerateToGpu(const SystemSampling& system, size_t beam_count = 1) override
```

**Параметры**:
- `system` — `const SystemSampling&`
- `beam_count` — `size_t`

**Возвращает**: `cl_mem`

**Doxygen-источник**:
```cpp
/**
     * @brief GPU production генерация CW (multi-beam через freq_step).
     *
     * @param system Параметры дискретизации (fs, length).
     * @param beam_count Количество лучей в выходе.
     *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
     *
     * @return cl_mem [beam_count × system.length × complex<float>]; caller обязан clReleaseMemObject.
     *   @test_check result != nullptr
     */
```

## Method 3: `GenerateToGpu`

**Сигнатура** (`cw_generator.hpp:112`):
```cpp
cl_mem GenerateToGpu(const SystemSampling& system, size_t beam_count, ProfEvents* prof_events)
```

**Параметры**:
- `system` — `const SystemSampling&`
- `beam_count` — `size_t`
- `prof_events` — `ProfEvents*` *(pointer)*

**Возвращает**: `cl_mem`

**Doxygen-источник**:
```cpp
/**
     * @brief Генерация на GPU с опциональным сбором событий профилирования.
     * @param prof_events nullptr → production (zero overhead); &vec → benchmark
     *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
     *
     * Собирает события: "Kernel" (cw_kernel)
     * @param system Параметры дискретизации (fs, length).
     * @param beam_count Количество лучей в выходе.
     *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
     * @return cl_mem [beam_count × system.length × complex<float>]; caller обязан clReleaseMemObject.
     *   @test_check result != nullptr
     */
```

## Method 4: `Kind`

**Сигнатура** (`cw_generator.hpp:122`):
```cpp
SignalKind Kind() const override { return SignalKind::CW;
```

**Возвращает**: `SignalKind`

**Doxygen-источник**:
```cpp
/**
     * @brief Возвращает тип сигнала (для introspection).
     *
     * @return Всегда `SignalKind::CW` для этого класса.
     *   @test_check result == SignalKind::CW
     */
```

