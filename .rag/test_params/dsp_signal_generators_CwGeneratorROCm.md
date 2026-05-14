---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::CwGeneratorROCm
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/cw_generator_rocm.hpp
line: 60
brief: "/**  * @class CwGeneratorROCm  * @brief ROCm/HIP-генератор CW (комплексной синусоиды) с поддержкой multi-beam.  *  * @ingroup grp_signal_generators  * @note Move-only: GpuContext + compiled module уни"
methods_total: 4
methods_with_doxygen: 4
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::CwGeneratorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class CwGeneratorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__cw_generator_rocm__class_overview__v1 -->

/**
 * @class CwGeneratorROCm
 * @brief ROCm/HIP-генератор CW (комплексной синусоиды) с поддержкой multi-beam.
 *
 * @ingroup grp_signal_generators
 * @note Move-only: GpuContext + compiled module уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note Доступен только при ENABLE_ROCM=1. OpenCL-вариант: CwGenerator.
 * @see dsp::signal_generators::CwGenerator
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->  # signal_generators — Repository Card  _Источник: `signal_generators/CLAUDE.md`_  # 🤖 CLAUDE — `signal_generators` …
- `signal_generators__cw_generator__class_overview__v1` (class_overview): /**  * @class CwGenerator  * @brief OpenCL-генератор CW (комплексной синусоиды) с поддержкой multi-beam.  *  * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.  * @note ba…

## Public-методы (4)

## Method 1: `GenerateToGpu`

**Сигнатура** (`cw_generator_rocm.hpp:85`):
```cpp
drv_gpu_lib::InputData<void*> GenerateToGpu( const SystemSampling& system, const CwParams& params, uint32_t beam_count, ROCmProfEvents* prof_events = nullptr)
```

**Параметры**:
- `system` — `const SystemSampling&`
- `params` — `const CwParams&`
- `beam_count` — `uint32_t`
- `prof_events` — `ROCmProfEvents*` *(pointer)*

**Возвращает**: `drv_gpu_lib::InputData<void*>`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация CW-сигнала на GPU
   * @param system     Параметры дискретизации (fs, length)
   * @param params     Параметры CW (f0, phase, amplitude, freq_step)
   *   @test_ref CwParams
   * @param beam_count Количество лучей
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   * @return InputData<void*> с GPU-сигналом (caller обязан hipFree result.data)
   *   @test_check result != nullptr
   */
```

## Method 2: `GenerateToCpu`

**Сигнатура** (`cw_generator_rocm.hpp:103`):
```cpp
std::vector<std::complex<float>> GenerateToCpu( const SystemSampling& system, const CwParams& params, uint32_t beam_count)
```

**Параметры**:
- `system` — `const SystemSampling&`
- `params` — `const CwParams&`
- `beam_count` — `uint32_t`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief CPU reference генерация CW (для сверки с GPU). Multi-beam через freq_step.
   *
   * @param system Параметры дискретизации (fs, length).
   * @param params Параметры CW (f0, phase, amplitude, freq_step).
   *   @test_ref CwParams
   * @param beam_count Количество лучей в выходе.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   *
   * @return Массив [beam_count × system.length] complex<float> (interleaved beams).
   *   @test_check result.size() == beam_count * system.length
   */
```

## Method 3: `GenerateToGpu`

**Сигнатура** (`cw_generator_rocm.hpp:141`):
```cpp
drv_gpu_lib::InputData<void*> GenerateToGpu(const SystemSampling&, const CwParams&, uint32_t, void* = nullptr) { throw std::runtime_error("CwGeneratorROCm: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const SystemSampling&`
- `_unnamed_` — `const CwParams&`
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

**Сигнатура** (`cw_generator_rocm.hpp:154`):
```cpp
std::vector<std::complex<float>> GenerateToCpu(const SystemSampling&, const CwParams&, uint32_t) { throw std::runtime_error("CwGeneratorROCm: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const SystemSampling&`
- `_unnamed_` — `const CwParams&`
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

