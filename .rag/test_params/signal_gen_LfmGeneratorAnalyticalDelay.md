---
schema_version: 1
repo: signal_generators
class_fqn: signal_gen::LfmGeneratorAnalyticalDelay
file: E:/DSP-GPU/signal_generators/include/signal_generators/generators/lfm_generator_analytical_delay.hpp
line: 49
brief: "Генератор LFM-сигналов с аналитической задержкой на антенну для GPU/CPU."
methods_total: 3
methods_with_doxygen: 3
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['lfm генератор с задержкой', 'аналитическая задержка', 'gpu сигнал генератор', 'многоантенный lfm']
synonyms_en: ['lfm delay generator', 'analytical delay', 'gpu signal generator', 'multi-antenna lfm']
tags: ['lfm', 'gpu', 'задержка', 'ROCm', 'signal_generation']
---

# `signal_gen::LfmGeneratorAnalyticalDelay` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class LfmGeneratorAnalyticalDelay`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__lfm_generator_analytical_delay__class_overview__v1 -->

**ЧТО**: Генератор LFM-сигналов с аналитической задержкой на антенну для GPU/CPU.

**ЗАЧЕМ**: Позволяет учитывать дробные задержки между антеннами в системах с множественными входами.

**КАК**: Использует аналитическую модель задержки для точного временного сдвига, оптимизирован для HIP/ROCm. Поддерживает синхронизацию с ProfEvents для профилирования.

**Пример**:
```cpp
#include "signal_generators/generators/lfm_generator_analytical_delay.hpp"

using namespace signal_gen;

int main() {
  IBackend* backend = new RocmBackend();
  LfmGeneratorAnalyticalDelay gen(backend);
  gen.SetParams({100e3, 1e6, 100});
  gen.SetSampling(SystemSampling::SAMPLING_100MHz);
  gen.SetDelays({0.0f, 0.27f, 0.54f});

  auto gpu_data = gen.GenerateToGpu();
  clReleaseMemObject(gpu_data.data);

  auto cpu_data = gen.GenerateToCpu();
  return 0;
}
```

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__gpu__c4_code__v1` (c4_code): ### C4 — Code (ключевые классы)  ``` FormSignalGenerator   + FormSignalGenerator(IBackend* backend)   + SetParams(const FormParams&)   + SetParamsFromString(const string&)   + GenerateInputData() → In…
- `signal_generators__gpu__section_002__v1` (section): ### Какой класс выбрать  | Класс | Для чего | Выход | Задержка | |-------|----------|-------|----------| | `FormSignalGenerator` | CW/Chirp + шум, N антенн | `InputData<cl_mem>` | FIXED/LINEAR/RANDOM …
- `signal_generators__gpu__section_003__v1` (section): ### Иерархия классов  ``` ISignalGenerator (interface)     ├── CwGenerator          → cw_kernel.cl     ├── LfmGenerator         → lfm_kernel.cl     └── NoiseGenerator       → noise_kernel.cl + prng.cl…

## Public-методы (3)

## Method 1: `GenerateToGpu`

**Сигнатура** (`lfm_generator_analytical_delay.hpp:107`):
```cpp
drv_gpu_lib::InputData<cl_mem> GenerateToGpu()
```

**Возвращает**: `drv_gpu_lib::InputData<cl_mem>`

**Doxygen-источник**:
```cpp
/**
   * @brief Generate on GPU
   * @return InputData<cl_mem> with [antennas * points] complex signal
   * @note Caller must release result.data via clReleaseMemObject()
   *   @test_check result.data != nullptr && result.antenna_count == delay_us_.size()
   */
```

## Method 2: `GenerateToGpu`

**Сигнатура** (`lfm_generator_analytical_delay.hpp:118`):
```cpp
drv_gpu_lib::InputData<cl_mem> GenerateToGpu(ProfEvents* prof_events)
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
   * Собирает события: "Kernel" (lfm_analytical_delay.cl)
   * @return InputData<cl_mem> [antennas × points × complex<float>]; caller обязан clReleaseMemObject result.data.
   *   @test_check result.data != nullptr
   */
```

## Method 3: `GenerateToCpu`

**Сигнатура** (`lfm_generator_analytical_delay.hpp:125`):
```cpp
std::vector<std::vector<std::complex<float>>> GenerateToCpu()
```

**Возвращает**: `std::vector<std::vector<std::complex<float>>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Generate on CPU (reference)
   * @return [antenna][sample] complex<float>
   *   @test_check result.size() == delay_us_.size() && result[0].size() == system_.length
   */
```

