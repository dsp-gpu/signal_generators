---
schema_version: 1
repo: signal_generators
class_fqn: dsp::signal_generators::ScriptGenerator
file: /home/alex/DSP-GPU/signal_generators/include/dsp/signal_generators/generators/script_generator.hpp
line: 90
brief: "/**  * @class ScriptGenerator  * @brief Компилятор текстового DSL в OpenCL kernel с исполнением на GPU.  *  * @note Move-only: cl_program/queue/context уникальны на инстанс.  * @note backend не владее"
methods_total: 4
methods_with_doxygen: 4
ai_generated: false
human_verified: false
parser_version: 1
---

# `dsp::signal_generators::ScriptGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class ScriptGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__script_generator__class_overview__v1 -->

/**
 * @class ScriptGenerator
 * @brief Компилятор текстового DSL в OpenCL kernel с исполнением на GPU.
 *
 * @note Move-only: cl_program/queue/context уникальны на инстанс.
 * @note backend не владеет — caller гарантирует переживание генератора.
 * @note OpenCL-вариант. ROCm-аналог: ScriptGeneratorROCm.
 * @see dsp::signal_generators::ScriptGeneratorROCm
 *
 * @code
 * ScriptGenerator gen(backend);
 * gen.LoadScript(R"(
 *     [Params]
 *     ANTENNAS = 8
 *     POINTS = 4096
 *     [Defs]
 *     var_W = 0.1 + (float)ID * 0.005
 *     [Signal]
 *     res = sin(var_W * (float)T);
 * )");
 *
 * cl_mem gpu_buf = gen.Generate();
 * // ... use gpu_buf ...
 * clReleaseMemObject(gpu_buf);
 * @endcode
 */

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->  # signal_generators — Repository Card  _Источник: `signal_generators/CLAUDE.md`_  # 🤖 CLAUDE — `signal_generators` …
- `signal_generators__architecture__section_002__v1` (section): ``` ┌─────────────────────────────────────────────────────────────┐ │                     Signal Generators                        │ ├─────────────────────────────────────────────────────────────┤ │  …
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__gpu__c4_code__v1` (c4_code): ### C4 — Code (ключевые классы)  ``` FormSignalGenerator   + FormSignalGenerator(IBackend* backend)   + SetParams(const FormParams&)   + SetParamsFromString(const string&)   + GenerateInputData() → In…

## Public-методы (4)

## Method 1: `LoadScript`

**Сигнатура** (`script_generator.hpp:108`):
```cpp
void LoadScript(const std::string& script_text)
```

**Параметры**:
- `script_text` — `const std::string&`

**Doxygen-источник**:
```cpp
/**
     * @brief Parse and compile script from string
     * @param script_text Script in [Params]/[Defs]/[Signal] format
     * @throws std::runtime_error on parse or compilation failure
     */
```

## Method 2: `LoadFile`

**Сигнатура** (`script_generator.hpp:116`):
```cpp
void LoadFile(const std::string& file_path)
```

**Параметры**:
- `file_path` — `const std::string&`

**Doxygen-источник**:
```cpp
/**
     * @brief Parse and compile script from file
     * @param file_path Path to .signal or .txt file
     *   @test { values=["/tmp/test_config.json"] }
     * @throws std::runtime_error if file cannot be read
     */
```

## Method 3: `Generate`

**Сигнатура** (`script_generator.hpp:124`):
```cpp
cl_mem Generate()
```

**Возвращает**: `cl_mem`

**Doxygen-источник**:
```cpp
/**
     * @brief Generate signal on GPU
     * @return cl_mem buffer [antennas * points * sizeof(complex<float>)]
     * @note Caller must release via clReleaseMemObject()
     *   @test_check result != nullptr (требуется LoadScript/LoadFile перед Generate)
     */
```

## Method 4: `GenerateToCpu`

**Сигнатура** (`script_generator.hpp:131`):
```cpp
std::vector<std::complex<float>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
     * @brief Generate and read back to CPU
     * @return Vector of complex samples [antennas * points]
     *   @test_check result.size() == script_.params.antennas * script_.params.points
     */
```

