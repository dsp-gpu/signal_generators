---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::NoiseGenerator
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/noise_generator.hpp
line: 56
brief: "/**  * @class NoiseGenerator  * @brief OpenCL-генератор Gaussian-шума (Philox-2x32 + Box-Muller).  *  * @note Move-only: GPU-ресурсы уникальны на инстанс.  * @note backend не владеет — caller гарантир"
methods_total: 4
methods_with_doxygen: 4
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::NoiseGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class NoiseGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__noise_generator__class_overview__v1 -->

/**
 * @class NoiseGenerator
 * @brief OpenCL-генератор Gaussian-шума (Philox-2x32 + Box-Muller).
 *
 * @note Move-only: GPU-ресурсы уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note Воспроизводимость через seed обязательна для unit-тестов.
 * @note OpenCL-вариант. ROCm-аналог: NoiseGeneratorROCm.
 * @see dsp::signal_generators::NoiseGeneratorROCm
 * @see dsp::signal_generators::ISignalGenerator
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->  # signal_generators — Repository Card  _Источник: `signal_generators/CLAUDE.md`_  # 🤖 CLAUDE — `signal_generators` …
- `signal_generators__architecture__section_002__v1` (section): ``` ┌─────────────────────────────────────────────────────────────┐ │                     Signal Generators                        │ ├─────────────────────────────────────────────────────────────┤ │  …
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__quick__tests__v1` (tests): ### Простые генераторы — для базовых тестов  **CwGenerator** — тон. Одна синусоида фиксированной частоты, один канал. Используй когда нужно проверить фильтр или FFT на простом предсказуемом сигнале.  …

## Public-методы (4)

## Method 1: `GenerateToCpu`

**Сигнатура** (`noise_generator.hpp:76`):
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
     * @brief CPU reference генерация шума (white или Gaussian по NoiseParams::type).
     *
     * @param system Параметры дискретизации (fs, length).
     * @param out Выходной буфер [out_size] complex<float>.
     * @param out_size Размер буфера (должен быть >= system.length).
     */
```

## Method 2: `GenerateToGpu`

**Сигнатура** (`noise_generator.hpp:89`):
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
     * @brief GPU production: Philox+BoxMuller kernel, multi-beam.
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

**Сигнатура** (`noise_generator.hpp:104`):
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
     * Собирает события: "Kernel" (noise_kernel / Philox+BoxMuller)
     * @param system Параметры дискретизации (fs, length).
     * @param beam_count Количество лучей в выходе.
     *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
     * @return cl_mem [beam_count × system.length × complex<float>]; caller обязан clReleaseMemObject.
     *   @test_check result != nullptr
     */
```

## Method 4: `Kind`

**Сигнатура** (`noise_generator.hpp:114`):
```cpp
SignalKind Kind() const override { return SignalKind::NOISE;
```

**Возвращает**: `SignalKind`

**Doxygen-источник**:
```cpp
/**
     * @brief Возвращает тип сигнала (для introspection).
     *
     * @return Всегда `SignalKind::NOISE` для этого класса.
     *   @test_check result == SignalKind::NOISE
     */
```

