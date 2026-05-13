---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::LfmConjugateGeneratorROCm
file: E:/DSP-GPU/signal_generators/include/signal_generators/generators/lfm_conjugate_generator_rocm.hpp
line: 34
brief: "Генерирует сигналы LFM сопряженного типа на GPU через ROCm."
methods_total: 4
methods_with_doxygen: 4
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['генератор LFM', 'сопряженный сигнал', 'ROCm-генератор', 'радиолокационный сигнал']
synonyms_en: ['LFM generator', 'conjugate signal', 'ROCm generator', 'radar signal']
tags: ['LFM', 'ROCm', 'генератор', 'сигнал', 'радиолокация']
---

# `dsp::signal_generators::LfmConjugateGeneratorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class LfmConjugateGeneratorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__lfm_conjugate_generator_rocm__class_overview__v1 -->

**ЧТО**: Генерирует сигналы LFM сопряженного типа на GPU через ROCm.

**ЗАЧЕМ**: Обеспечивает оптимизацию радиолокационной обработки сигналов на GPU с поддержкой ROCm.

**КАК**: Использует lazy init для отложенной инициализации, кэширование данных и асинхронные паттерны для GPU-вычислений. Методы выбрасывают исключение при отсутствии ROCm.

**Пример**:
```cpp
#include <dsp/signal_generators/generators/lfm_conjugate_generator_rocm.hpp>

using namespace dsp::signal_generators;

int main() {
    LfmConjugateGeneratorROCm gen;
    try {
        auto gpu_data = gen.GenerateToGpu();
        // Обработка GPU-данных
    } catch (const std::runtime_error& e) {
        // Обработка ошибки ROCm
    }
    return 0;
}
```

<!-- /rag-block -->

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

