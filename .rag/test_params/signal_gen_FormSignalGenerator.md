---
schema_version: 1
repo: signal_generators
class_fqn: signal_gen::FormSignalGenerator
file: E:/DSP-GPU/signal_generators/include/signal_generators/generators/form_signal_generator.hpp
line: 67
brief: "Генератор комплексных сигналов на GPU по пользовательской формуле с поддержкой мультиканального формирования."
methods_total: 3
methods_with_doxygen: 3
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['Генератор сигналов', 'Формирование сигналов', 'GPU-генератор', 'Мультиканальный генератор']
synonyms_en: ['Signal generator', 'Signal formation', 'GPU generator', 'Multi-channel generator']
tags: ['GPU', 'сигналы', 'мультиканал', 'ROCm', 'HIP']
---

# `signal_gen::FormSignalGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class FormSignalGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__form_signal_generator__class_overview__v1 -->

**ЧТО**: Генератор комплексных сигналов на GPU по пользовательской формуле с поддержкой мультиканального формирования.

**ЗАЧЕМ**: Решает задачу создания гибких сигналов с точностью до 1e-3 для радиолокационной обработки и FFT-анализа.

**КАК**: Использует lazy init для ядра, кэширование параметров, поддержку batch-генерации и оптимизацию для ROCm/HIP. Разделяет GPU/CPU выводы.

**Пример**:
```cpp
#include "signal_generators/generators/form_signal_generator.hpp"
using namespace signal_gen;

int main() {
  auto backend = drv_gpu_lib::GetDefaultBackend();
  FormSignalGenerator gen(backend);

  FormParams params;
  params.fs = 12e6;
  params.f0 = 1e6;
  params.antennas = 8;
  params.points = 4096;
  gen.SetParams(params);

  auto input = gen.GenerateInputData();
  // ... передать в SpectrumMaximaFinder::Process(input)
  clReleaseMemObject(input.data);

  auto cpu_data = gen.GenerateToCpu();
  return 0;
}
```

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__tests_003__v1` (tests): | # | ID | Файл | Что проверяет | Параметры | Порог | |---|----|------|---------------|-----------|-------| | 1 | SG-1 | test_signal_generators.hpp | CW: GPU vs CPU | f0=250 Hz, A=1.5, fs=4 kHz, N=409…
- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->  # signal_generators — Repository Card  _Источник: `signal_generators/CLAUDE.md`_  # 🤖 CLAUDE — `signal_generators` …
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__gpu__tests_005__v1` (tests): ### 7.2 Python тесты  | # | Файл | Функция | Что проверяет | Порог | |---|------|---------|---------------|-------| | 1 | test_form_signal.py | test_no_noise | FormSignal vs NumPy getX | max_err < 1e-…

## Public-методы (3)

## Method 1: `GenerateInputData`

**Сигнатура** (`form_signal_generator.hpp:94`):
```cpp
drv_gpu_lib::InputData<cl_mem> GenerateInputData()
```

**Возвращает**: `drv_gpu_lib::InputData<cl_mem>`

**Doxygen-источник**:
```cpp
/**
   * @brief Генерация на GPU с метаданными (InputData — как в fft_func)
   * @return InputData<cl_mem> с data, antenna_count, n_point, gpu_memory_bytes
   * @note Вызывающий код должен освободить input.data через clReleaseMemObject()!
   *   @test_check result.data != nullptr && result.antenna_count == params_.antennas
   */
```

## Method 2: `GenerateInputData`

**Сигнатура** (`form_signal_generator.hpp:105`):
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
   * Собирает события: "Kernel" (form_signal.cl)
   * @return InputData<cl_mem> с data, antenna_count, n_point, gpu_memory_bytes.
   *   @test_check result.data != nullptr && result.antenna_count == params_.antennas
   */
```

## Method 3: `GenerateToCpu`

**Сигнатура** (`form_signal_generator.hpp:112`):
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

