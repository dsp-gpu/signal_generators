---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::NoiseGeneratorROCm
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/noise_generator_rocm.hpp
line: 62
brief: "/**  * @class NoiseGeneratorROCm  * @brief ROCm/HIP-генератор гауссовского комплексного шума.  *  * @note Move-only: GPU-ресурсы (GpuContext, hipModule) уникальны.  * @note Требует #if ENABLE_ROCM. На"
methods_total: 4
methods_with_doxygen: 4
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::NoiseGeneratorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class NoiseGeneratorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__noise_generator_rocm__class_overview__v1 -->

/**
 * @class NoiseGeneratorROCm
 * @brief ROCm/HIP-генератор гауссовского комплексного шума.
 *
 * @note Move-only: GPU-ресурсы (GpuContext, hipModule) уникальны.
 * @note Требует #if ENABLE_ROCM. На Windows — stub (все методы throw).
 * @note API совместим с NoiseGenerator (OpenCL) по семантике.
 * @see dsp::signal_generators::NoiseGenerator (legacy OpenCL)
 * @see drv_gpu_lib::GpuContext (Layer 1 Ref03)
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->  # signal_generators — Repository Card  _Источник: `signal_generators/CLAUDE.md`_  # 🤖 CLAUDE — `signal_generators` …
- `signal_generators__noise_generator__class_overview__v1` (class_overview): /**  * @class NoiseGenerator  * @brief OpenCL-генератор Gaussian-шума (Philox-2x32 + Box-Muller).  *  * @note Move-only: GPU-ресурсы уникальны на инстанс.  * @note backend не владеет — caller гарантир…

## Public-методы (4)

## Method 1: `GenerateToGpu`

**Сигнатура** (`noise_generator_rocm.hpp:86`):
```cpp
drv_gpu_lib::InputData<void*> GenerateToGpu( const SystemSampling& system, const NoiseParams& params, uint32_t beam_count, ROCmProfEvents* prof_events = nullptr)
```

**Параметры**:
- `system` — `const SystemSampling&`
- `params` — `const NoiseParams&`
- `beam_count` — `uint32_t`
- `prof_events` — `ROCmProfEvents*` *(pointer)*

**Возвращает**: `drv_gpu_lib::InputData<void*>`

**Doxygen-источник**:
```cpp
/**
   * @brief GPU production генерация шума через Philox+BoxMuller. Multi-beam за один launch.
   *
   * @param system Параметры дискретизации (fs, length).
   * @param params Параметры шума (type, power, seed).
   *   @test_ref NoiseParams
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

**Сигнатура** (`noise_generator_rocm.hpp:104`):
```cpp
std::vector<std::complex<float>> GenerateToCpu( const SystemSampling& system, const NoiseParams& params, uint32_t beam_count)
```

**Параметры**:
- `system` — `const SystemSampling&`
- `params` — `const NoiseParams&`
- `beam_count` — `uint32_t`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief CPU reference генерация шума (для unit-тестов и сверки с GPU).
   *
   * @param system Параметры дискретизации (fs, length).
   * @param params Параметры шума (type, power, seed).
   *   @test_ref NoiseParams
   * @param beam_count Количество лучей в выходе.
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   *
   * @return Массив [beam_count × system.length] complex<float> (interleaved beams).
   *   @test_check result.size() == beam_count * system.length
   */
```

## Method 3: `GenerateToGpu`

**Сигнатура** (`noise_generator_rocm.hpp:143`):
```cpp
drv_gpu_lib::InputData<void*> GenerateToGpu(const SystemSampling&, const NoiseParams&, uint32_t, void* = nullptr) { throw std::runtime_error("NoiseGeneratorROCm: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const SystemSampling&`
- `_unnamed_` — `const NoiseParams&`
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

**Сигнатура** (`noise_generator_rocm.hpp:156`):
```cpp
std::vector<std::complex<float>> GenerateToCpu(const SystemSampling&, const NoiseParams&, uint32_t) { throw std::runtime_error("NoiseGeneratorROCm: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const SystemSampling&`
- `_unnamed_` — `const NoiseParams&`
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

