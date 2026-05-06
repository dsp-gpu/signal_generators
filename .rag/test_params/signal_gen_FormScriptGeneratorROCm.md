---
schema_version: 1
repo: signal_generators
class_fqn: signal_gen::FormScriptGeneratorROCm
file: E:/DSP-GPU/signal_generators/include/signal_generators/generators/form_script_generator_rocm.hpp
line: 28
brief: "Класс предоставляет интерфейс для генерации входных данных для GPU-обработки сигналов с поддержкой ROCm."
methods_total: 4
methods_with_doxygen: 4
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['генератор данных', 'ROCm', 'GPU-обработка', 'сигнал', 'входные данные']
synonyms_en: ['data generator', 'ROCm', 'GPU processing', 'signal', 'input data']
tags: ['GPU', 'ROCm', 'генератор', 'сигнал', 'данные']
---

# `signal_gen::FormScriptGeneratorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class FormScriptGeneratorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__form_script_generator_rocm__class_overview__v1 -->

**ЧТО**: Класс предоставляет интерфейс для генерации входных данных для GPU-обработки сигналов с поддержкой ROCm.

**ЗАЧЕМ**: Решает проблему платформенной зависимости генерации данных для GPU, обеспечивая отдельную реализацию для ROCm.

**КАК**: Использует паттерн lazy initialization для проверки активации ROCm. Все методы выбрасывают исключение при отсутствии поддержки. Реализует абстрактные методы для разных GPU-платформ.

**Пример**:
```cpp
#include "signal_generators/generators/form_script_generator_rocm.hpp"
using namespace signal_generators;

int main() {
    FormScriptGeneratorROCm gen;
    try {
        auto input = gen.GenerateInputData();
        auto cpu_data = gen.GenerateToCpu();
    } catch (const std::runtime_error& e) {
        // Обработка ошибки отсутствия ROCm
    }
    return 0;
}
```

<!-- /rag-block -->

## Public-методы (4)

## Method 1: `GenerateInputData`

**Сигнатура** (`form_script_generator_rocm.hpp:72`):
```cpp
drv_gpu_lib::InputData<void*> GenerateInputData()
```

**Возвращает**: `drv_gpu_lib::InputData<void*>`

**Doxygen-источник**:
```cpp
/**
   * @brief GPU production: FormParams → DSL → hiprtc → HIP launch. Возвращает InputData<void*>.
   *
   * @return InputData<void*> [antennas × points × complex<float>]; caller обязан hipFree result.data.
   *   @test_check result != nullptr && result.antenna_count == params_.antennas
   */
```

## Method 2: `GenerateToCpu`

**Сигнатура** (`form_script_generator_rocm.hpp:80`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::vector<std::complex<float>>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Полный pipeline с readback на CPU (для unit-тестов и сверки).
   *
   * @return vector[antenna_id][sample_id] complex<float>.
   *   @test_check result.size() == params_.antennas && result[0].size() == params_.points
   */
```

## Method 3: `GenerateInputData`

**Сигнатура** (`form_script_generator_rocm.hpp:124`):
```cpp
drv_gpu_lib::InputData<void*> GenerateInputData() { throw std::runtime_error("FormScriptGeneratorROCm: ROCm not enabled");
```

**Возвращает**: `drv_gpu_lib::InputData<void*>`

**Doxygen-источник**:
```cpp
/**
   * @brief Stub: бросает runtime_error — GenerateInputData доступен только в ROCm-сборке.
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
```

## Method 4: `GenerateToCpu`

**Сигнатура** (`form_script_generator_rocm.hpp:136`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu() { throw std::runtime_error("FormScriptGeneratorROCm: ROCm not enabled");
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

