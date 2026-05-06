---
schema_version: 1
repo: signal_generators
class_fqn: signal_gen::LfmConjugateGenerator
file: E:/DSP-GPU/signal_generators/include/signal_generators/generators/lfm_conjugate_generator.hpp
line: 48
brief: "Генерирует конъюгированный LFM-сигнал для деширпирования"
methods_total: 3
methods_with_doxygen: 3
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['генератор конъюгированного LFM', 'опорный сигнал для деширпа', 'деширп-генератор', 'сопряженный LFM-сигнал']
synonyms_en: ['conjugate LFM generator', 'dechirp reference signal', 'dechirp generator', 'conjugated LFM signal']
tags: ['GPU', 'CPU', 'деширпирование', 'LFM', 'сигналы']
---

# `signal_gen::LfmConjugateGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class LfmConjugateGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__lfm_conjugate_generator__class_overview__v1 -->

**ЧТО**: Генерирует конъюгированный LFM-сигнал для деширпирования

**ЗАЧЕМ**: Позволяет создавать опорный сигнал для обработки отраженных волн в радарных системах

**КАК**: Оптимизирован для GPU с direct memory access, поддерживает lazy init и кэширование промежуточных данных

**Пример**:
```cpp
#include "signal_generators/generators/lfm_conjugate_generator.hpp"

using namespace signal_gen;

int main() {
  auto gen = LfmConjugateGenerator::Create(Backend::HIP, LfmParams{...});
  gen->SetSampling(SystemSampling::SAMPLING_1);
  
  cl_mem gpu_ref = gen->GenerateToGpu();
  // ... обработка в деширп-пайплайне ...
  clReleaseMemObject(gpu_ref);

  auto cpu_ref = gen->GenerateToCpu();
  // ... обработка на CPU ...
  return 0;
}
```

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__gpu__section_002__v1` (section): ### Какой класс выбрать  | Класс | Для чего | Выход | Задержка | |-------|----------|-------|----------| | `FormSignalGenerator` | CW/Chirp + шум, N антенн | `InputData<cl_mem>` | FIXED/LINEAR/RANDOM …
- `signal_generators__gpu__section_003__v1` (section): ### Иерархия классов  ``` ISignalGenerator (interface)     ├── CwGenerator          → cw_kernel.cl     ├── LfmGenerator         → lfm_kernel.cl     └── NoiseGenerator       → noise_kernel.cl + prng.cl…
- `signal_generators__quick__section_002__v1` (section): ## Какой класс выбрать  | Задача | Класс | |--------|-------| | CW/Chirp + шум, N каналов | `FormSignalGenerator` | | То же + on-disk кэш kernel | `FormScriptGenerator` | | Дробная задержка (Farrow 48…

## Public-методы (3)

## Method 1: `GenerateToGpu`

**Сигнатура** (`lfm_conjugate_generator.hpp:100`):
```cpp
cl_mem GenerateToGpu()
```

**Возвращает**: `cl_mem`

**Doxygen-источник**:
```cpp
/**
   * @brief Generate conjugate LFM on GPU
   * @return cl_mem with [num_samples] complex signal (conj LFM)
   * @note Caller must release via clReleaseMemObject()
   *   @test_check result != nullptr (cl_mem [system_.length × complex<float>])
   */
```

## Method 2: `GenerateToGpu`

**Сигнатура** (`lfm_conjugate_generator.hpp:111`):
```cpp
cl_mem GenerateToGpu(ProfEvents* prof_events)
```

**Параметры**:
- `prof_events` — `ProfEvents*` *(pointer)*

**Возвращает**: `cl_mem`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация на GPU с опциональным сбором событий профилирования.
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * Собирает события: "Kernel" (lfm_conjugate.cl)
   * @return cl_mem [system_.length × complex<float>] с conj(LFM); caller обязан clReleaseMemObject.
   *   @test_check result != nullptr
   */
```

## Method 3: `GenerateToCpu`

**Сигнатура** (`lfm_conjugate_generator.hpp:118`):
```cpp
std::vector<std::complex<float>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Generate conjugate LFM on CPU (reference)
   * @return vector of complex<float>, length = system_.length
   *   @test_check result.size() == system_.length
   */
```

