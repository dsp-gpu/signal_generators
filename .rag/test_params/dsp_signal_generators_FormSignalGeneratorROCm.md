---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::FormSignalGeneratorROCm
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/form_signal_generator_rocm.hpp
line: 66
brief: "/**  * @class FormSignalGeneratorROCm  * @brief ROCm/HIP-генератор getX (мультиканал, встроенный Philox-шум, ЛЧМ).  *  * @ingroup grp_signal_generators  * @note Move-only: GpuContext + compiled module"
methods_total: 5
methods_with_doxygen: 5
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::FormSignalGeneratorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class FormSignalGeneratorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__form_signal_generator_rocm__class_overview__v1 -->

/**
 * @class FormSignalGeneratorROCm
 * @brief ROCm/HIP-генератор getX (мультиканал, встроенный Philox-шум, ЛЧМ).
 *
 * @ingroup grp_signal_generators
 * @note Move-only: GpuContext + compiled module уникальны на инстанс.
 * @note Доступен только при ENABLE_ROCM=1. OpenCL-вариант: FormSignalGenerator.
 * @see dsp::signal_generators::FormSignalGenerator
 * @see dsp::signal_generators::DelayedFormSignalGeneratorROCm
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__gpu__s_6_10_python_formsignalgenerator__v1` (s_6_10_python_formsignalgenerator): ### 6.10 Python — FormSignalGenerator  ```python import dsp_signal_generators import numpy as np  ctx = dsp_signal_generators.ROCmGPUContext(0) gen = dsp_signal_generators.FormSignalGeneratorROCm(ctx)…
- `signal_generators__delayed_form_signal_generator_rocm__class_overview__v1` (class_overview): /**  * @class DelayedFormSignalGeneratorROCm  * @brief ROCm-композит: FormSignalGeneratorROCm + LchFarrowROCm (delay + noise).  *  * @ingroup grp_signal_generators  * @note Доступен только при ENABLE_…
- `signal_generators__delayed_form_signal_generator__class_overview__v1` (class_overview): /**  * @class DelayedFormSignalGenerator  * @brief OpenCL-генератор getX с дробной задержкой Farrow (Lagrange 48×5).  *  * @note Move-only (composite над move-only компонентами).  * @note Шум идёт ПОС…
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

