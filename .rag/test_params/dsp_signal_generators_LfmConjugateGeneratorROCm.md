---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::LfmConjugateGeneratorROCm
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/lfm_conjugate_generator_rocm.hpp
line: 60
brief: "/**  * @class LfmConjugateGeneratorROCm  * @brief ROCm/HIP conjugate-LFM генератор для dechirp-обработки.  *  * @note Move-only: GPU-ресурсы (GpuContext, hipModule) уникальны.  * @note Требует #if ENA"
methods_total: 4
methods_with_doxygen: 4
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::LfmConjugateGeneratorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class LfmConjugateGeneratorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__lfm_conjugate_generator_rocm__class_overview__v1 -->

/**
 * @class LfmConjugateGeneratorROCm
 * @brief ROCm/HIP conjugate-LFM генератор для dechirp-обработки.
 *
 * @note Move-only: GPU-ресурсы (GpuContext, hipModule) уникальны.
 * @note Требует #if ENABLE_ROCM. На Windows — stub (все методы throw).
 * @note API совместим с LfmConjugateGenerator (OpenCL).
 * @see dsp::signal_generators::LfmConjugateGenerator (legacy OpenCL)
 * @see drv_gpu_lib::GpuContext (Layer 1 Ref03)
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__lfm_conjugate_generator__class_overview__v1` (class_overview): /**  * @class LfmConjugateGenerator  * @brief GPU/CPU conjugate-LFM генератор — reference для dechirp.  *  * @note Move-only: cl_program/queue/context уникальны на инстанс.  * @note backend не владеет…

## Public-методы (4)

## Method 1: `GenerateToGpu`

**Сигнатура** (`lfm_conjugate_generator_rocm.hpp:82`):
```cpp
void* GenerateToGpu()
```

**Doxygen-источник**:
```cpp
/**
   * @brief Generate conjugate LFM on GPU (ROCm)
   * @return HIP device pointer [num_samples × complex<float>]
   *         CALLER OWNS — must hipFree!
   */
```

## Method 2: `GenerateToCpu`

**Сигнатура** (`lfm_conjugate_generator_rocm.hpp:89`):
```cpp
std::vector<std::complex<float>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Generate conjugate LFM on CPU (reference)
   * @return vector<complex<float>>, length = system_.length
   *   @test_check result.size() == system_.length
   */
```

## Method 3: `GenerateToGpu`

**Сигнатура** (`lfm_conjugate_generator_rocm.hpp:125`):
```cpp
void* GenerateToGpu() { throw std::runtime_error("LfmConjugateGeneratorROCm: ROCm not enabled");
```

**Doxygen-источник**:
```cpp
/**
   * @brief Stub: бросает runtime_error — GenerateToGpu доступен только в ROCm-сборке.
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
```

## Method 4: `GenerateToCpu`

**Сигнатура** (`lfm_conjugate_generator_rocm.hpp:135`):
```cpp
std::vector<std::complex<float>> GenerateToCpu() { throw std::runtime_error("LfmConjugateGeneratorROCm: ROCm not enabled");
```

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Stub: бросает runtime_error — GenerateToCpu доступен только в ROCm-сборке.
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
```

