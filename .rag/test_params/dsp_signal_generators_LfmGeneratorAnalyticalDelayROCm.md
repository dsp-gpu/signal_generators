---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::LfmGeneratorAnalyticalDelayROCm
file: E:/DSP-GPU/signal_generators/include/signal_generators/generators/lfm_generator_analytical_delay_rocm.hpp
line: 35
brief: "Генератор сигналов с линейной модуляцией (LFM) с аналитической задержкой, использующий ROCm для GPU-вычислений."
methods_total: 4
methods_with_doxygen: 4
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['Генератор LFM с задержкой на ROCm', 'Аналитический генератор сигналов ROCm', 'Сигнал генератор LFM GPU', 'ROCm LFM-сигнал']
synonyms_en: ['ROCm LFM Signal Generator', 'Analytical Delay LFM Generator', 'GPU LFM Signal Generator', 'ROCm LFM Generator']
tags: ['GPU', 'ROCm', 'LFM', 'Сигналы', 'Радиолокация']
---

# `dsp::signal_generators::LfmGeneratorAnalyticalDelayROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class LfmGeneratorAnalyticalDelayROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__lfm_generator_analytical_delay_rocm__class_overview__v1 -->

**ЧТО**: Генератор сигналов с линейной модуляцией (LFM) с аналитической задержкой, использующий ROCm для GPU-вычислений.

**ЗАЧЕМ**: Решает задачу эффективной генерации LFM-сигналов на GPU для радиолокационных систем с поддержкой ROCm.

**КАК**: Реализует асинхронную генерацию сигналов на GPU с кэшированием результатов. Поддерживает CPU-вычисления как fallback. Использует паттерн lazy init для оптимизации памяти.

**Пример**:
```cpp
#include "dsp/signal_generators/generators/lfm_generator_analytical_delay_rocm.hpp"

using namespace dsp::signal_generators;

int main() {
    LfmGeneratorAnalyticalDelayROCm gen;
    auto gpu_data = gen.GenerateToGpu();
    auto cpu_data = gen.GenerateToCpu();
    return 0;
}
```

<!-- /rag-block -->

## Public-методы (4)

## Method 1: `GenerateToGpu`

**Сигнатура** (`lfm_generator_analytical_delay_rocm.hpp:89`):
```cpp
drv_gpu_lib::InputData<void*> GenerateToGpu(ROCmProfEvents* prof_events = nullptr)
```

**Параметры**:
- `prof_events` — `ROCmProfEvents*` *(pointer)*

**Возвращает**: `drv_gpu_lib::InputData<void*>`

**Doxygen-источник**:
```cpp
/**
   * @brief GPU production генерация LFM с аналитической per-antenna задержкой.
   *
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * @return InputData<void*> [antennas × points × complex<float>]; caller обязан hipFree result.data.
   *   @test_check result != nullptr
   */
```

## Method 2: `GenerateToCpu`

**Сигнатура** (`lfm_generator_analytical_delay_rocm.hpp:97`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::vector<std::complex<float>>>`

**Doxygen-источник**:
```cpp
/**
   * @brief CPU reference генерация LFM с аналитической задержкой (для сверки с GPU).
   *
   * @return vector[antenna_id][sample_id] complex<float>.
   *   @test_check result.size() == delay_us_.size() && result[0].size() == system_.length
   */
```

## Method 3: `GenerateToGpu`

**Сигнатура** (`lfm_generator_analytical_delay_rocm.hpp:138`):
```cpp
drv_gpu_lib::InputData<void*> GenerateToGpu(void* = nullptr) { throw std::runtime_error("LfmGeneratorAnalyticalDelayROCm: ROCm not enabled");
```

**Параметры**:
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

**Сигнатура** (`lfm_generator_analytical_delay_rocm.hpp:150`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu() { throw std::runtime_error("LfmGeneratorAnalyticalDelayROCm: ROCm not enabled");
```

**Возвращает**: `std::vector<std::vector<std::complex<float>>>`

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

