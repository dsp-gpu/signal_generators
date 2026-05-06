---
schema_version: 1
repo: signal_generators
class_fqn: signal_gen::DelayedFormSignalGeneratorROCm
file: E:/DSP-GPU/signal_generators/include/signal_generators/generators/delayed_form_signal_generator_rocm.hpp
line: 31
brief: "Генерирует сигналы с задержками для радиолокационной обработки на GPU с использованием ROCm."
methods_total: 5
methods_with_doxygen: 5
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['Генератор задержек', 'Сигнал ROCm', 'Генератор форм']
synonyms_en: ['Delayed Signal Generator', 'ROCm Signal Generator', 'Form Signal Generator']
tags: ['GPU', 'ROCm', 'Сигналы', 'Задержки', 'Обработка сигналов']
---

# `signal_gen::DelayedFormSignalGeneratorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class DelayedFormSignalGeneratorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__delayed_form_signal_generator_rocm__class_overview__v1 -->

**ЧТО**: Генерирует сигналы с задержками для радиолокационной обработки на GPU с использованием ROCm.

**ЗАЧЕМ**: Решает проблему эффективной генерации задержанных сигналов на GPU для ускорения обработки в реальном времени.

**КАК**: Использует ROCm для GPU-вычислений, кэширование матриц задержек, поддержка батч-генерации. Методы с проверкой на активацию ROCm.

**Пример**:
```cpp
#include "signal_generators/generators/delayed_form_signal_generator_rocm.hpp"
using namespace signal_generators;

DelayedFormSignalGeneratorROCm gen;
gen.LoadMatrix("params.json");
gen.set_params(fs=1e6, f0=50000, antennas=8, points=4096);
gen.set_delays({0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5});
auto data = gen.GenerateToCpu();
```

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__s_6_11_python_delayedformsignalgenerator__v1` (s_6_11_python_delayedformsignalgenerator): ### 6.11 Python — DelayedFormSignalGenerator  ```python gen = dsp_signal_generators.DelayedFormSignalGeneratorROCm(ctx) gen.set_params(     fs=1e6, f0=50000,     antennas=8, points=4096,     amplitude…
- `signal_generators__quick__python_delayedformsignalgenerator__v1` (python_delayedformsignalgenerator): ### Python — DelayedFormSignalGenerator  ```python gen = dsp_signal_generators.DelayedFormSignalGeneratorROCm(ctx) gen.set_params(fs=1e6, f0=50000, antennas=4, points=4096) gen.set_delays([0.0, 1.5, 3…

## Public-методы (5)

## Method 1: `LoadMatrix`

**Сигнатура** (`delayed_form_signal_generator_rocm.hpp:78`):
```cpp
void LoadMatrix(const std::string& json_path) { lch_farrow_.LoadMatrix(json_path);
```

**Параметры**:
- `json_path` — `const std::string&`

**Doxygen-источник**:
```cpp
/**

   * @brief Загружает матрицу Lagrange 48×5 для Farrow-фильтра из JSON-файла.

   *

   * @param json_path Путь к JSON с матрицей (формат: { "data": [[...], ...] }).

   *   @test { values=["/tmp/test_config.json"] }

   */
```

## Method 2: `GenerateInputData`

**Сигнатура** (`delayed_form_signal_generator_rocm.hpp:91`):
```cpp
drv_gpu_lib::InputData<void*> GenerateInputData()
```

**Возвращает**: `drv_gpu_lib::InputData<void*>`

**Doxygen-источник**:
```cpp
/**

   * @brief GPU production: чистый сигнал → задержка Farrow → шум. Возвращает InputData<void*>.

   *

   * @return InputData<void*> [antennas × points × complex<float>]; caller обязан hipFree result.data.

   *   @test_check result != nullptr && result.antenna_count == params_.antennas

   */
```

## Method 3: `GenerateToCpu`

**Сигнатура** (`delayed_form_signal_generator_rocm.hpp:99`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::vector<std::complex<float>>>`

**Doxygen-источник**:
```cpp
/**

   * @brief Полный pipeline с readback на CPU (для unit-тестов и сверки с GPU).

   *

   * @return vector[antenna_id][sample_id] complex<float>.

   *   @test_check result.size() == params_.antennas && result[0].size() == params_.points

   */
```

## Method 4: `GenerateInputData`

**Сигнатура** (`delayed_form_signal_generator_rocm.hpp:134`):
```cpp
drv_gpu_lib::InputData<void*> GenerateInputData() { throw std::runtime_error("DelayedFormSignalGeneratorROCm: ROCm not enabled");
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

## Method 5: `GenerateToCpu`

**Сигнатура** (`delayed_form_signal_generator_rocm.hpp:146`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu() { throw std::runtime_error("DelayedFormSignalGeneratorROCm: ROCm not enabled");
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


## Python API

**Pybind модуль**: `dsp_signal_generators` · **Класс Python**: `DelayedFormSignalGeneratorROCm` · **Wrapper C++**: `PyDelayedFormSignalGeneratorROCm`

_Источник биндинга_: `signal_generators/python/py_delayed_form_signal_rocm.hpp`

**Конструктор**: `py::init<ROCmGPUContext&>()`

| Python | C++ | Overload |
|---|---|---|
| `set_params` | `PyDelayedFormSignalGeneratorROCm::set_params` | — |
| `set_delays` | `PyDelayedFormSignalGeneratorROCm::set_delays` | — |
| `load_matrix` | `PyDelayedFormSignalGeneratorROCm::load_matrix` | — |
| `generate` | `PyDelayedFormSignalGeneratorROCm::generate` | — |
| `get_params` | `PyDelayedFormSignalGeneratorROCm::get_params_dict` | — |
