---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::LfmGeneratorAnalyticalDelay
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/lfm_generator_analytical_delay.hpp
line: 70
brief: "/**  * @class LfmGeneratorAnalyticalDelay  * @brief GPU/CPU LFM-генератор с аналитической per-antenna задержкой.  *  * @note Move-only: cl_program/queue/context уникальны на инстанс.  * @note backend"
methods_total: 3
methods_with_doxygen: 3
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::LfmGeneratorAnalyticalDelay` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class LfmGeneratorAnalyticalDelay`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__lfm_generator_analytical_delay__class_overview__v1 -->

/**
 * @class LfmGeneratorAnalyticalDelay
 * @brief GPU/CPU LFM-генератор с аналитической per-antenna задержкой.
 *
 * @note Move-only: cl_program/queue/context уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note OpenCL-вариант. ROCm-аналог: LfmGeneratorAnalyticalDelayROCm.
 * @see dsp::signal_generators::LfmGeneratorAnalyticalDelayROCm
 *
 * @code
 * LfmGeneratorAnalyticalDelay gen(backend);
 * gen.SetParams(lfm_params);
 * gen.SetSampling(system);
 * gen.SetDelays({0.0f, 0.27f, 0.54f});  // microseconds
 *
 * // GPU
 * auto result = gen.GenerateToGpu();
 * clReleaseMemObject(result.data);
 *
 * // CPU
 * auto cpu = gen.GenerateToCpu();
 * @endcode
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__gpu__c4_code__v1` (c4_code): ### C4 — Code (ключевые классы)  ``` FormSignalGenerator   + FormSignalGenerator(IBackend* backend)   + SetParams(const FormParams&)   + SetParamsFromString(const string&)   + GenerateInputData() → In…
- `signal_generators__gpu__s_4_3_lfmgeneratoranalyticaldelay__v1` (s_4_3_lfmgeneratoranalyticaldelay): ### 4.3 LfmGeneratorAnalyticalDelay  ``` INPUT: LfmParams + SystemSampling + delay_us[] per-antenna     │     ▼ ┌──────────────────────────────────────────────────┐ │ 1. lfm_analytical_delay kernel (G…
- `signal_generators__gpu__s_6_6_c_lfmgeneratoranalyticaldelay__v1` (s_6_6_c_lfmgeneratoranalyticaldelay): ### 6.6 C++ — LfmGeneratorAnalyticalDelay  ```cpp #include "generators/lfm_generator_analytical_delay.hpp"  dsp::signal_generators::LfmParams params; params.f_start = 1e6;  params.f_end = 2e6;  params…

## Public-методы (3)

## Method 1: `GenerateToGpu`

**Сигнатура** (`lfm_generator_analytical_delay.hpp:107`):
```cpp
drv_gpu_lib::InputData<cl_mem> GenerateToGpu()
```

**Возвращает**: `drv_gpu_lib::InputData<cl_mem>`

**Doxygen-источник**:
```cpp
/**
   * @brief Generate on GPU
   * @return InputData<cl_mem> with [antennas * points] complex signal
   * @note Caller must release result.data via clReleaseMemObject()
   *   @test_check result.data != nullptr && result.antenna_count == delay_us_.size()
   */
```

## Method 2: `GenerateToGpu`

**Сигнатура** (`lfm_generator_analytical_delay.hpp:118`):
```cpp
drv_gpu_lib::InputData<cl_mem> GenerateToGpu(ProfEvents* prof_events)
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
   * Собирает события: "Kernel" (lfm_analytical_delay.cl)
   * @return InputData<cl_mem> [antennas × points × complex<float>]; caller обязан clReleaseMemObject result.data.
   *   @test_check result.data != nullptr
   */
```

## Method 3: `GenerateToCpu`

**Сигнатура** (`lfm_generator_analytical_delay.hpp:125`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::vector<std::complex<float>>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Generate on CPU (reference)
   * @return [antenna][sample] complex<float>
   *   @test_check result.size() == delay_us_.size() && result[0].size() == system_.length
   */
```

