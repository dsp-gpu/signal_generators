---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::LfmGenerator
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/lfm_generator.hpp
line: 52
brief: "/**  * @class LfmGenerator  * @brief OpenCL-генератор LFM chirp с поддержкой multi-beam.  *  * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.  * @note backend не владеет"
methods_total: 4
methods_with_doxygen: 4
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::LfmGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class LfmGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__lfm_generator__class_overview__v1 -->

/**
 * @class LfmGenerator
 * @brief OpenCL-генератор LFM chirp с поддержкой multi-beam.
 *
 * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note Доступен только в OpenCL-сборке. ROCm-вариант: LfmGeneratorROCm.
 * @see dsp::signal_generators::LfmGeneratorROCm
 * @see dsp::signal_generators::ISignalGenerator
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__tests_003__v1` (tests): | # | ID | Файл | Что проверяет | Параметры | Порог | |---|----|------|---------------|-----------|-------| | 1 | SG-1 | test_signal_generators.hpp | CW: GPU vs CPU | f0=250 Hz, A=1.5, fs=4 kHz, N=409…
- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->  # signal_generators — Repository Card  _Источник: `signal_generators/CLAUDE.md`_  # 🤖 CLAUDE — `signal_generators` …
- `signal_generators__architecture__section_002__v1` (section): ``` ┌─────────────────────────────────────────────────────────────┐ │                     Signal Generators                        │ ├─────────────────────────────────────────────────────────────┤ │  …
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…

## Public-методы (4)

## Method 1: `GenerateToCpu`

**Сигнатура** (`lfm_generator.hpp:72`):
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
     * @brief CPU reference генерация LFM chirp.
     *
     * @param system Параметры дискретизации (fs, length).
     * @param out Выходной буфер [out_size] complex<float>.
     * @param out_size Размер буфера (должен быть >= system.length).
     */
```

## Method 2: `GenerateToGpu`

**Сигнатура** (`lfm_generator.hpp:85`):
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
     * @brief GPU production генерация LFM chirp (multi-beam).
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

**Сигнатура** (`lfm_generator.hpp:100`):
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
     * Собирает события: "Kernel" (lfm_kernel)
     * @param system Параметры дискретизации (fs, length).
     * @param beam_count Количество лучей в выходе.
     *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
     * @return cl_mem [beam_count × system.length × complex<float>]; caller обязан clReleaseMemObject.
     *   @test_check result != nullptr
     */
```

## Method 4: `Kind`

**Сигнатура** (`lfm_generator.hpp:110`):
```cpp
SignalKind Kind() const override { return SignalKind::LFM;
```

**Возвращает**: `SignalKind`

**Doxygen-источник**:
```cpp
/**
     * @brief Возвращает тип сигнала (для introspection).
     *
     * @return Всегда `SignalKind::LFM` для этого класса.
     *   @test_check result == SignalKind::LFM
     */
```

