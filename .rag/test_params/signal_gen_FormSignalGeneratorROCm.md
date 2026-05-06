---
schema_version: 1
repo: signal_generators
class_fqn: signal_gen::FormSignalGeneratorROCm
file: E:/DSP-GPU/signal_generators/include/signal_generators/generators/form_signal_generator_rocm.hpp
line: 52
brief: "Генерирует задержанные сигналы для радиолокационной обработки на GPU с использованием ROCm"
methods_total: 5
methods_with_doxygen: 5
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['Генератор сигналов ROCm', 'ROCm Signal Generator', 'Задержанный сигнал', 'Delayed Signal Generator', 'GPU Signal Generator', 'Form Signal Generator']
synonyms_en: ['ROCm Signal Generator', 'Delayed Signal Generator', 'GPU Signal Generator', 'Form Signal Generator', 'Signal Generator ROCm', 'Delayed Signal']
tags: ['ROCm', 'GPU', 'Signal Generation', 'Delayed Signal', 'Real-time Processing']
---

# `signal_gen::FormSignalGeneratorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class FormSignalGeneratorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__form_signal_generator_rocm__class_overview__v1 -->

**ЧТО**: Генерирует задержанные сигналы для радиолокационной обработки на GPU с использованием ROCm

**ЗАЧЕМ**: Решает проблему эффективной генерации задержанных сигналов на GPU для ускорения обработки в реальном времени

**КАК**: Использует ROCm для GPU-вычислений, кэширует матрицы задержек, поддерживает батч-генерацию. Методы проверяют активацию ROCm

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

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__delayed_form_signal_generator_rocm__class_overview__v1` (class_overview): **ЧТО**: Генерирует сигналы с задержками для радиолокационной обработки на GPU с использованием ROCm.  **ЗАЧЕМ**: Решает проблему эффективной генерации задержанных сигналов на GPU для ускорения обрабо…
- `signal_generators__gpu__section_003__v1` (section): ### Иерархия классов  ``` ISignalGenerator (interface)     ├── CwGenerator          → cw_kernel.cl     ├── LfmGenerator         → lfm_kernel.cl     └── NoiseGenerator       → noise_kernel.cl + prng.cl…
- `signal_generators__gpu__s_6_10_python_formsignalgenerator__v1` (s_6_10_python_formsignalgenerator): ### 6.10 Python — FormSignalGenerator  ```python import dsp_signal_generators import numpy as np  ctx = dsp_signal_generators.ROCmGPUContext(0) gen = dsp_signal_generators.FormSignalGeneratorROCm(ctx)…
- `signal_generators__quick__python_formsignalgenerator__v1` (python_formsignalgenerator): ### Python — FormSignalGenerator  ```python import dsp_signal_generators  ctx = dsp_signal_generators.ROCmGPUContext(0) gen = dsp_signal_generators.FormSignalGeneratorROCm(ctx)  gen.set_params(fs=12e6…

## Public-методы (5)

## Method 1: `GenerateInputData`

**Сигнатура** (`form_signal_generator_rocm.hpp:97`):
```cpp
drv_gpu_lib::InputData<void*> GenerateInputData()
```

**Возвращает**: `drv_gpu_lib::InputData<void*>`

**Doxygen-источник**:
```cpp
/**

   * @brief Генерация сигнала на GPU (ROCm)

   * @return InputData<void*> с сгенерированным сигналом (caller обязан hipFree result.data)

   *   @test_check result != nullptr

   */
```

## Method 2: `GenerateInputData`

**Сигнатура** (`form_signal_generator_rocm.hpp:108`):
```cpp
drv_gpu_lib::InputData<void*> GenerateInputData(ROCmProfEvents* prof_events)
```

**Параметры**:
- `prof_events` — `ROCmProfEvents*` *(pointer)*

**Возвращает**: `drv_gpu_lib::InputData<void*>`

**Doxygen-источник**:
```cpp
/**

   * @brief Генерация на GPU с опциональным сбором событий профилирования (ROCm)

   * @param prof_events nullptr → production (zero overhead); &vec → benchmark

   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }

   *

   * Собирает события: "Kernel" (generate_form_signal HIP kernel)

   * @return InputData<void*> с HIP device pointer; caller обязан hipFree result.data.

   *   @test_check result != nullptr

   */
```

## Method 3: `GenerateToCpu`

**Сигнатура** (`form_signal_generator_rocm.hpp:115`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::vector<std::complex<float>>>`

**Doxygen-источник**:
```cpp
/**

   * @brief Генерация на GPU с возвратом результата на CPU

   * @return vector[antenna][sample] = complex<float>

   *   @test_check result.size() == params_.antennas && result[0].size() == params_.points

   */
```

## Method 4: `GenerateInputData`

**Сигнатура** (`form_signal_generator_rocm.hpp:177`):
```cpp
drv_gpu_lib::InputData<void*> GenerateInputData() { throw std::runtime_error("FormSignalGeneratorROCm: ROCm not enabled");
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

**Сигнатура** (`form_signal_generator_rocm.hpp:190`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu() { throw std::runtime_error("FormSignalGeneratorROCm: ROCm not enabled");
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

**Pybind модуль**: `dsp_signal_generators` · **Класс Python**: `FormSignalGeneratorROCm` · **Wrapper C++**: `PyFormSignalGeneratorROCm`

_Источник биндинга_: `signal_generators/python/py_form_signal_rocm.hpp`

**Конструктор**: `py::init<ROCmGPUContext&>()`

| Python | C++ | Overload |
|---|---|---|
| `set_params` | `PyFormSignalGeneratorROCm::set_params` | — |
| `set_params_from_string` | `PyFormSignalGeneratorROCm::set_params_from_string` | — |
| `generate` | `PyFormSignalGeneratorROCm::generate` | — |
| `get_params` | `PyFormSignalGeneratorROCm::get_params_dict` | — |
