---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::FormScriptGenerator
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/form_script_generator.hpp
line: 80
brief: "/**  * @class FormScriptGenerator  * @brief DSL-генератор + on-disk кэш OpenCL kernels для формулы getX.  *  * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.  * @note То"
methods_total: 9
methods_with_doxygen: 9
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::FormScriptGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class FormScriptGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__form_script_generator__class_overview__v1 -->

/**
 * @class FormScriptGenerator
 * @brief DSL-генератор + on-disk кэш OpenCL kernels для формулы getX.
 *
 * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.
 * @note Только OpenCL. ROCm-аналог: FormScriptGeneratorROCm.
 * @see dsp::signal_generators::FormScriptGeneratorROCm
 * @see dsp::signal_generators::FormSignalGenerator
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__gpu__c4_code__v1` (c4_code): ### C4 — Code (ключевые классы)  ``` FormSignalGenerator   + FormSignalGenerator(IBackend* backend)   + SetParams(const FormParams&)   + SetParamsFromString(const string&)   + GenerateInputData() → In…
- `signal_generators__gpu__s_6_8_c_formscriptgenerator__v1` (s_6_8_c_formscriptgenerator): ### 6.8 C++ — FormScriptGenerator  ```cpp #include "generators/form_script_generator.hpp"  dsp::signal_generators::FormScriptGenerator gen(backend); gen.SetParams(p);  // Режим 1: компиляция + сохране…
- `signal_generators__gpu__s_6_13_python_formscriptgenerator__v1` (s_6_13_python_formscriptgenerator): ### 6.13 Python — FormScriptGenerator  ```python gen = dsp_signal_generators.FormScriptGenerator(ctx) gen.set_params(fs=10e6, f0=1e6, antennas=16, points=8192, noise_amplitude=0.05) gen.compile() gen.…

## Public-методы (9)

## Method 1: `GenerateScript`

**Сигнатура** (`form_script_generator.hpp:114`):
```cpp
std::string GenerateScript() const
```

**Возвращает**: `std::string`

**Doxygen-источник**:
```cpp
/**
   * @brief Возвращает читаемый DSL-скрипт (текстовое представление getX-формулы по params_).
   *
   * @return Multi-line строка с DSL-описанием сигнала (Params/Defs/Signal секции).
   *   @test_check !result.empty()
   */
```

## Method 2: `GenerateKernelSource`

**Сигнатура** (`form_script_generator.hpp:122`):
```cpp
std::string GenerateKernelSource() const
```

**Возвращает**: `std::string`

**Doxygen-источник**:
```cpp
/**
   * @brief Возвращает полный OpenCL kernel source (с PRNG-helpers и `#define`-параметрами).
   *
   * @return Готовый к clBuildProgram OpenCL C исходник.
   *   @test_check !result.empty()
   */
```

## Method 3: `Compile`

**Сигнатура** (`form_script_generator.hpp:127`):
```cpp
void Compile()
```

**Doxygen-источник**:
```cpp
/**
   * @brief Компилирует OpenCL kernel из текущих params (внутри: GenerateKernelSource → clBuildProgram).
   */
```

## Method 4: `SaveKernel`

**Сигнатура** (`form_script_generator.hpp:141`):
```cpp
void SaveKernel(const std::string& name, const std::string& comment = "")
```

**Параметры**:
- `name` — `const std::string&`
- `comment` — `const std::string&`

**Doxygen-источник**:
```cpp
/**
   * @brief Сохранить текущий kernel на диск
   * @param name Имя (без расширения): "work_sig0"
   * @param comment Комментарий (записывается в manifest)
   *
   * Создаёт: name.cl + bin/name_opencl.bin
   * При коллизии: старые файлы → name_00.cl, name_opencl_00.bin
   */
```

## Method 5: `LoadKernel`

**Сигнатура** (`form_script_generator.hpp:149`):
```cpp
void LoadKernel(const std::string& name)
```

**Параметры**:
- `name` — `const std::string&`

**Doxygen-источник**:
```cpp
/**
   * @brief Загрузить kernel с диска по имени
   * @param name Имя кернела (без расширения)
   *
   * Приоритет: binary (fast) → source (compile + save binary)
   */
```

## Method 6: `ListKernels`

**Сигнатура** (`form_script_generator.hpp:157`):
```cpp
std::vector<std::string> ListKernels() const
```

**Возвращает**: `std::vector<std::string>`

**Doxygen-источник**:
```cpp
/**
   * @brief Возвращает список имён кернелов из on-disk manifest.json.
   *
   * @return vector<string> с именами (без расширений); пуст если кэш ещё не создан.
   *   @test_check result.size() >= 0 (количество ранее сохранённых через SaveKernel)
   */
```

## Method 7: `GenerateInputData`

**Сигнатура** (`form_script_generator.hpp:169`):
```cpp
drv_gpu_lib::InputData<cl_mem> GenerateInputData()
```

**Возвращает**: `drv_gpu_lib::InputData<cl_mem>`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация на GPU с метаданными (InputData — как в fft_func)
   * @return InputData<cl_mem> с data, antenna_count, n_point, gpu_memory_bytes
   * @note Вызывающий код должен освободить input.data через clReleaseMemObject()
   *   @test_check result.data != nullptr && result.antenna_count == params_.antennas
   */
```

## Method 8: `GenerateInputData`

**Сигнатура** (`form_script_generator.hpp:180`):
```cpp
drv_gpu_lib::InputData<cl_mem> GenerateInputData(ProfEvents* prof_events)
```

**Параметры**:
- `prof_events` — `ProfEvents*` *(pointer)*

**Возвращает**: `drv_gpu_lib::InputData<cl_mem>`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация на GPU с опциональным сбором событий профилирования.
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * Собирает события: "Kernel" (form_script_signal kernel)
   * @return InputData<cl_mem> с data, antenna_count, n_point, gpu_memory_bytes; caller обязан clReleaseMemObject result.data.
   *   @test_check result.data != nullptr
   */
```

## Method 9: `GenerateToCpu`

**Сигнатура** (`form_script_generator.hpp:187`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::vector<std::complex<float>>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация с возвратом на CPU (по каналам)
   * @return vector[antenna_id][sample_id] complex<float>
   *   @test_check result.size() == params_.antennas && result[0].size() == params_.points
   */
```

