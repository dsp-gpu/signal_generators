---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::DelayedFormSignalGenerator
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/delayed_form_signal_generator.hpp
line: 69
brief: "/**  * @class DelayedFormSignalGenerator  * @brief OpenCL-генератор getX с дробной задержкой Farrow (Lagrange 48×5).  *  * @note Move-only (composite над move-only компонентами).  * @note Шум идёт ПОС"
methods_total: 4
methods_with_doxygen: 4
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::DelayedFormSignalGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class DelayedFormSignalGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__delayed_form_signal_generator__class_overview__v1 -->

/**
 * @class DelayedFormSignalGenerator
 * @brief OpenCL-генератор getX с дробной задержкой Farrow (Lagrange 48×5).
 *
 * @note Move-only (composite над move-only компонентами).
 * @note Шум идёт ПОСЛЕ задержки (через LchFarrow::SetNoise), не до.
 * @note Только OpenCL. ROCm-аналог: DelayedFormSignalGeneratorROCm.
 * @see dsp::signal_generators::DelayedFormSignalGeneratorROCm
 * @see dsp::signal_generators::FormSignalGenerator
 * @see dsp::spectrum::LchFarrow
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__gpu__c4_code__v1` (c4_code): ### C4 — Code (ключевые классы)  ``` FormSignalGenerator   + FormSignalGenerator(IBackend* backend)   + SetParams(const FormParams&)   + SetParamsFromString(const string&)   + GenerateInputData() → In…
- `signal_generators__gpu__s_4_2_delayedformsignalgenerator__v1` (s_4_2_delayedformsignalgenerator): ### 4.2 DelayedFormSignalGenerator  ``` INPUT: FormParams + delay_us[] (задержки в мкс per-antenna)     │     ▼ ┌──────────────────────────────────────────────────┐ │ 1. FormSignalGenerator::GenerateI…
- `signal_generators__gpu__s_6_5_c_delayedformsignalgenerator__v1` (s_6_5_c_delayedformsignalgenerator): ### 6.5 C++ — DelayedFormSignalGenerator  ```cpp #include "generators/delayed_form_signal_generator.hpp"  dsp::signal_generators::FormParams p; p.fs = 12e6;  p.f0 = 1e6; p.antennas = 8;  p.points = 40…

## Public-методы (4)

## Method 1: `LoadMatrix`

**Сигнатура** (`delayed_form_signal_generator.hpp:104`):
```cpp
void LoadMatrix(const std::string& json_path)
```

**Параметры**:
- `json_path` — `const std::string&`

**Doxygen-источник**:
```cpp
/**
   * @brief Загрузить матрицу Lagrange из JSON-файла
   * @param json_path Путь к файлу (формат: { "data": [[...], ...] })
   *   @test { values=["/tmp/test_config.json"] }
   *
   * Если не вызвано — используется встроенная матрица 48×5 из LchFarrow.
   */
```

## Method 2: `GenerateInputData`

**Сигнатура** (`delayed_form_signal_generator.hpp:112`):
```cpp
drv_gpu_lib::InputData<cl_mem> GenerateInputData()
```

**Возвращает**: `drv_gpu_lib::InputData<cl_mem>`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация на GPU: сигнал + задержка + шум
   * @return InputData<cl_mem>
   * @note Вызывающий код освобождает result.data через clReleaseMemObject()
   *   @test_check result.data != nullptr && result.antenna_count == params.antennas
   */
```

## Method 3: `GenerateInputData`

**Сигнатура** (`delayed_form_signal_generator.hpp:123`):
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
   * Собирает события: "Kernel" (FormSignal), "Upload_delay", "Kernel" (FarrowDelay)
   * @return InputData<cl_mem> с задержанным FormSignal; caller обязан clReleaseMemObject result.data.
   *   @test_check result.data != nullptr && result.antenna_count == params.antennas
   */
```

## Method 4: `GenerateToCpu`

**Сигнатура** (`delayed_form_signal_generator.hpp:130`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::vector<std::complex<float>>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация с возвратом на CPU
   * @return vector[antenna_id][sample_id] complex<float>
   *   @test_check result.size() == params.antennas && result[0].size() == params.points
   */
```

