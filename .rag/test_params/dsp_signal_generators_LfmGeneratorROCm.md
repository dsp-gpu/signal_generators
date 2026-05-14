---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::LfmGeneratorROCm
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/lfm_generator_rocm.hpp
line: 59
brief: "/**  * @class LfmGeneratorROCm  * @brief ROCm/HIP-генератор LFM chirp с multi-beam.  *  * @note Move-only: GPU-ресурсы (GpuContext, hipModule) уникальны.  * @note Требует #if ENABLE_ROCM. На Windows —"
methods_total: 4
methods_with_doxygen: 4
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::LfmGeneratorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class LfmGeneratorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__lfm_generator_rocm__class_overview__v1 -->

/**
 * @class LfmGeneratorROCm
 * @brief ROCm/HIP-генератор LFM chirp с multi-beam.
 *
 * @note Move-only: GPU-ресурсы (GpuContext, hipModule) уникальны.
 * @note Требует #if ENABLE_ROCM. На Windows — stub (все методы throw).
 * @note API совместим с LfmGenerator для прозрачной замены backend'а.
 * @see dsp::signal_generators::LfmGenerator (legacy OpenCL)
 * @see drv_gpu_lib::GpuContext (Layer 1 Ref03)
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->  # signal_generators — Repository Card  _Источник: `signal_generators/CLAUDE.md`_  # 🤖 CLAUDE — `signal_generators` …
- `signal_generators__lfm_generator__class_overview__v1` (class_overview): /**  * @class LfmGenerator  * @brief OpenCL-генератор LFM chirp с поддержкой multi-beam.  *  * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.  * @note backend не владеет…

## Public-методы (4)

## Method 1: `GenerateToGpu`

**Сигнатура** (`lfm_generator_rocm.hpp:83`):
```cpp
drv_gpu_lib::InputData<void*> GenerateToGpu( const SystemSampling& system, const LfmParams& params, uint32_t beam_count, ROCmProfEvents* prof_events = nullptr)
```

**Параметры**:
- `system` — `const SystemSampling&`
- `params` — `const LfmParams&`
- `beam_count` — `uint32_t`
- `prof_events` — `ROCmProfEvents*` *(pointer)*

**Возвращает**: `drv_gpu_lib::InputData<void*>`

**Doxygen-источник**:
```cpp
/**
   * @brief GPU production генерация LFM chirp. Multi-beam за один HIP launch.
   *
   * @param system Параметры дискретизации (fs, length).
   * @param params Параметры LFM (f_start, f_end, amplitude, complex_iq).
   *   @test_ref LfmParams
   * @param beam_count Количество лучей в выходе.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * @return InputData<void*> с HIP device pointer; caller обязан hipFree result.data.
   *   @test_check result != nullptr
   */
```

## Method 2: `GenerateToCpu`

**Сигнатура** (`lfm_generator_rocm.hpp:101`):
```cpp
std::vector<std::complex<float>> GenerateToCpu( const SystemSampling& system, const LfmParams& params, uint32_t beam_count)
```

**Параметры**:
- `system` — `const SystemSampling&`
- `params` — `const LfmParams&`
- `beam_count` — `uint32_t`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief CPU reference генерация LFM (для unit-тестов и сверки с GPU).
   *
   * @param system Параметры дискретизации (fs, length).
   * @param params Параметры LFM (f_start, f_end, amplitude, complex_iq).
   *   @test_ref LfmParams
   * @param beam_count Количество лучей в выходе.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   *
   * @return Массив [beam_count × system.length] complex<float> (interleaved beams).
   *   @test_check result.size() == beam_count * system.length
   */
```

## Method 3: `GenerateToGpu`

**Сигнатура** (`lfm_generator_rocm.hpp:139`):
```cpp
drv_gpu_lib::InputData<void*> GenerateToGpu(const SystemSampling&, const LfmParams&, uint32_t, void* = nullptr) { throw std::runtime_error("LfmGeneratorROCm: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const SystemSampling&`
- `_unnamed_` — `const LfmParams&`
- `_unnamed_` — `uint32_t`
- `_unnamed_` — `void*` *(pointer)* *(void\*)*

**Возвращает**: `drv_gpu_lib::InputData<void*>`

**Doxygen-источник**:
```cpp
/**
   * @brief Stub: бросает runtime_error — GenerateToGpu доступен только в ROCm-сборке.
   *
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
```

## Method 4: `GenerateToCpu`

**Сигнатура** (`lfm_generator_rocm.hpp:152`):
```cpp
std::vector<std::complex<float>> GenerateToCpu(const SystemSampling&, const LfmParams&, uint32_t) { throw std::runtime_error("LfmGeneratorROCm: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const SystemSampling&`
- `_unnamed_` — `const LfmParams&`
- `_unnamed_` — `uint32_t`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Stub: бросает runtime_error — GenerateToCpu доступен только в ROCm-сборке.
   *
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
```

