---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::FormSignalGenerator
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/form_signal_generator.hpp
line: 64
brief: "/**  * @class FormSignalGenerator  * @brief OpenCL-генератор комплексных сигналов по формуле getX (мультиканал).  *  * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.  *"
methods_total: 3
methods_with_doxygen: 3
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::FormSignalGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class FormSignalGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__form_signal_generator__class_overview__v1 -->

/**
 * @class FormSignalGenerator
 * @brief OpenCL-генератор комплексных сигналов по формуле getX (мультиканал).
 *
 * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.
 * @note Доступен только в OpenCL-сборке. ROCm-вариант: FormSignalGeneratorROCm.
 * @see dsp::signal_generators::FormSignalGeneratorROCm
 * @see dsp::signal_generators::DelayedFormSignalGenerator
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__tests_003__v1` (tests): | # | ID | Файл | Что проверяет | Параметры | Порог | |---|----|------|---------------|-----------|-------| | 1 | SG-1 | test_signal_generators.hpp | CW: GPU vs CPU | f0=250 Hz, A=1.5, fs=4 kHz, N=409…
- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->  # signal_generators — Repository Card  _Источник: `signal_generators/CLAUDE.md`_  # 🤖 CLAUDE — `signal_generators` …
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__gpu__tests_005__v1` (tests): ### 7.2 Python тесты  | # | Файл | Функция | Что проверяет | Порог | |---|------|---------|---------------|-------| | 1 | test_form_signal.py | test_no_noise | FormSignal vs NumPy getX | max_err < 1e-…

## Public-методы (3)

## Method 1: `GenerateInputData`

**Сигнатура** (`form_signal_generator.hpp:94`):
```cpp
drv_gpu_lib::InputData<cl_mem> GenerateInputData()
```

**Возвращает**: `drv_gpu_lib::InputData<cl_mem>`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация на GPU с метаданными (InputData — как в fft_func)
   * @return InputData<cl_mem> с data, antenna_count, n_point, gpu_memory_bytes
   * @note Вызывающий код должен освободить input.data через clReleaseMemObject()!
   *   @test_check result.data != nullptr && result.antenna_count == params_.antennas
   */
```

## Method 2: `GenerateInputData`

**Сигнатура** (`form_signal_generator.hpp:105`):
```cpp
drv_gpu_lib::InputData<cl_mem> GenerateInputData(ProfEvents* prof_events)
```

**Параметры**:
- `prof_events` — `ProfEvents*` *(pointer)*

**Возвращает**: `drv_gpu_lib::InputData<cl_mem>`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация на GPU с опциональным сбором событий профилирования.
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * Собирает события: "Kernel" (form_signal.cl)
   * @return InputData<cl_mem> с data, antenna_count, n_point, gpu_memory_bytes.
   *   @test_check result.data != nullptr && result.antenna_count == params_.antennas
   */
```

## Method 3: `GenerateToCpu`

**Сигнатура** (`form_signal_generator.hpp:112`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::vector<std::complex<float>>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация с возвратом на CPU (по каналам)
   * @return vector[antenna_id][sample_id] complex<float>
   *   @test_check result.size() == params_.antennas && result[0].size() == params_.points
   */
```

