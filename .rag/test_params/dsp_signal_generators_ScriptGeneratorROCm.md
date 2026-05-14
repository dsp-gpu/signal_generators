---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::ScriptGeneratorROCm
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/script_generator_rocm.hpp
line: 65
brief: "/**  * @class ScriptGeneratorROCm  * @brief ROCm/HIP DSL-генератор с disk-cache HSACO через GpuContext.  *  * @note Move-only: GpuContext (unique_ptr) и hipStream уникальны.  * @note Требует #if ENABL"
methods_total: 7
methods_with_doxygen: 7
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::ScriptGeneratorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class ScriptGeneratorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__script_generator_rocm__class_overview__v1 -->

/**
 * @class ScriptGeneratorROCm
 * @brief ROCm/HIP DSL-генератор с disk-cache HSACO через GpuContext.
 *
 * @note Move-only: GpuContext (unique_ptr) и hipStream уникальны.
 * @note Требует #if ENABLE_ROCM. На Windows — stub (все методы throw).
 * @note Per-script GpuContext: каждый LoadScript пересоздаёт контекст
 *       (см. ctx_) — disk cache ключуется CompileKey.Hash.
 * @see dsp::signal_generators::ScriptGenerator (legacy OpenCL)
 * @see drv_gpu_lib::GpuContext (Layer 1 Ref03)
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__script_generator__class_overview__v1` (class_overview): /**  * @class ScriptGenerator  * @brief Компилятор текстового DSL в OpenCL kernel с исполнением на GPU.  *  * @note Move-only: cl_program/queue/context уникальны на инстанс.  * @note backend не владее…
- `signal_generators__form_script_generator__class_overview__v1` (class_overview): /**  * @class FormScriptGenerator  * @brief DSL-генератор + on-disk кэш OpenCL kernels для формулы getX.  *  * @note Move-only: GPU-ресурсы (cl_program/queue/context) уникальны на инстанс.  * @note То…
- `signal_generators__form_script_generator_rocm__class_overview__v1` (class_overview): /**  * @class FormScriptGeneratorROCm  * @brief Композиция над ScriptGeneratorROCm: FormParams → DSL → HIP-сигнал.  *  * @ingroup grp_signal_generators  * @note Доступен только при ENABLE_ROCM=1. Open…
- `signal_generators__form_script_generator_rocm__method_generatetocpu_signature_002__v1` (method_generatetocpu_signature): ```cpp std::vector<std::vector<std::complex<float>>> GenerateToCpu() { throw std::runtime_error("FormScriptGeneratorROCm: ROCm not enabled"); ```…
- `signal_generators__form_script_generator_rocm__method_generateinputdata_signature_002__v1` (method_generateinputdata_signature): ```cpp drv_gpu_lib::InputData<void*> GenerateInputData() { throw std::runtime_error("FormScriptGeneratorROCm: ROCm not enabled"); ```…

## Public-методы (7)

## Method 1: `LoadScript`

**Сигнатура** (`script_generator_rocm.hpp:80`):
```cpp
void LoadScript(const std::string& script_text)
```

**Параметры**:
- `script_text` — `const std::string&`

**Doxygen-источник**:
```cpp
/**
   * @brief Парсит DSL-скрипт и компилирует HIP kernel через GpuContext (с disk cache).
   *
   * @param script_text DSL-текст ([Params]/[Defs]/[Signal] секции).
   */
```

## Method 2: `LoadFile`

**Сигнатура** (`script_generator_rocm.hpp:87`):
```cpp
void LoadFile(const std::string& file_path)
```

**Параметры**:
- `file_path` — `const std::string&`

**Doxygen-источник**:
```cpp
/**
   * @brief Загружает DSL-скрипт из файла и компилирует kernel.
   *
   * @param file_path Путь к .signal/.txt файлу.
   *   @test { values=["/tmp/test_config.json"] }
   */
```

## Method 3: `Generate`

**Сигнатура** (`script_generator_rocm.hpp:92`):
```cpp
void* Generate()
```

**Doxygen-источник**:
```cpp
/**
   * @brief Запускает скомпилированный kernel на GPU; возвращает device pointer (caller обязан hipFree).
   */
```

## Method 4: `GenerateToCpu`

**Сигнатура** (`script_generator_rocm.hpp:100`):
```cpp
std::vector<std::complex<float>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Полный pipeline с readback на CPU (для unit-тестов и сверки).
   *
   * @return Массив [antennas × points] complex<float>.
   *   @test_check result.size() == antennas_ * points_
   */
```

## Method 5: `LoadScript`

**Сигнатура** (`script_generator_rocm.hpp:154`):
```cpp
void LoadScript(const std::string&) { throw std::runtime_error("ScriptGeneratorROCm: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const std::string&`

**Doxygen-источник**:
```cpp
/**
   * @brief Stub: бросает runtime_error — LoadScript доступен только в ROCm-сборке.
   *
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
```

## Method 6: `Generate`

**Сигнатура** (`script_generator_rocm.hpp:161`):
```cpp
void* Generate() { throw std::runtime_error("ScriptGeneratorROCm: ROCm not enabled");
```

**Doxygen-источник**:
```cpp
/**
   * @brief Stub: бросает runtime_error — Generate доступен только в ROCm-сборке.
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
```

## Method 7: `GenerateToCpu`

**Сигнатура** (`script_generator_rocm.hpp:171`):
```cpp
std::vector<std::complex<float>> GenerateToCpu() { throw std::runtime_error("ScriptGeneratorROCm: ROCm not enabled");
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

