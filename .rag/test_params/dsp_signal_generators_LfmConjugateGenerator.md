---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::LfmConjugateGenerator
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/lfm_conjugate_generator.hpp
line: 69
brief: "/**  * @class LfmConjugateGenerator  * @brief GPU/CPU conjugate-LFM генератор — reference для dechirp.  *  * @note Move-only: cl_program/queue/context уникальны на инстанс.  * @note backend не владеет"
methods_total: 3
methods_with_doxygen: 3
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::LfmConjugateGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class LfmConjugateGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__lfm_conjugate_generator__class_overview__v1 -->

/**
 * @class LfmConjugateGenerator
 * @brief GPU/CPU conjugate-LFM генератор — reference для dechirp.
 *
 * @note Move-only: cl_program/queue/context уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note OpenCL-вариант. ROCm-аналог: LfmConjugateGeneratorROCm.
 * @see dsp::signal_generators::LfmConjugateGeneratorROCm
 * @see HeterodyneDechirp (radar/heterodyne)
 *
 * @code
 * LfmConjugateGenerator gen(backend, lfm_params);
 * gen.SetSampling(system);
 *
 * // GPU
 * cl_mem ref = gen.GenerateToGpu();
 * // ... use in dechirp pipeline ...
 * clReleaseMemObject(ref);
 *
 * // CPU
 * auto ref_cpu = gen.GenerateToCpu();
 * @endcode
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__quick__section_002__v1` (section): ## Какой класс выбрать  | Задача | Класс | |--------|-------| | CW/Chirp + шум, N каналов | `FormSignalGenerator` | | То же + on-disk кэш kernel | `FormScriptGenerator` | | Дробная задержка (Farrow 48…
- `signal_generators__gpu__section_006__v1` (section): ### Ссылки  | Документ | Описание | |----------|----------| | [Quick.md](Quick.md) | Краткий справочник — первый старт | | [Doc/Modules/lch_farrow/Full.md](../lch_farrow/Full.md) | Математика Farrow 4…
- `signal_generators__quick__lfmconjugategenerator__v1` (lfmconjugategenerator): ### LfmConjugateGenerator — опорный сигнал для дечирпа  **Что делает**: генерирует комплексно сопряжённую копию LFM сигнала — conj(s_tx). Это не имитация принятого сигнала, а опорный сигнал для операц…

## Public-методы (3)

## Method 1: `GenerateToGpu`

**Сигнатура** (`lfm_conjugate_generator.hpp:100`):
```cpp
cl_mem GenerateToGpu()
```

**Возвращает**: `cl_mem`

**Doxygen-источник**:
```cpp
/**
   * @brief Generate conjugate LFM on GPU
   * @return cl_mem with [num_samples] complex signal (conj LFM)
   * @note Caller must release via clReleaseMemObject()
   *   @test_check result != nullptr (cl_mem [system_.length × complex<float>])
   */
```

## Method 2: `GenerateToGpu`

**Сигнатура** (`lfm_conjugate_generator.hpp:111`):
```cpp
cl_mem GenerateToGpu(ProfEvents* prof_events)
```

**Параметры**:
- `prof_events` — `ProfEvents*` *(pointer)*

**Возвращает**: `cl_mem`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация на GPU с опциональным сбором событий профилирования.
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * Собирает события: "Kernel" (lfm_conjugate.cl)
   * @return cl_mem [system_.length × complex<float>] с conj(LFM); caller обязан clReleaseMemObject.
   *   @test_check result != nullptr
   */
```

## Method 3: `GenerateToCpu`

**Сигнатура** (`lfm_conjugate_generator.hpp:118`):
```cpp
std::vector<std::complex<float>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Generate conjugate LFM on CPU (reference)
   * @return vector of complex<float>, length = system_.length
   *   @test_check result.size() == system_.length
   */
```

