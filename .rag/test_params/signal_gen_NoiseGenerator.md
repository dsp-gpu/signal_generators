---
schema_version: 1
repo: signal_generators
class_fqn: signal_gen::NoiseGenerator
file: E:/DSP-GPU/signal_generators/include/signal_generators/generators/noise_generator.hpp
line: 24
brief: "Генерирует белый шум с использованием rocRAND или Philox PRNG на CPU/GPU"
methods_total: 4
methods_with_doxygen: 4
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['шум', 'белый шум', 'гауссовский шум', 'генератор шума', 'прерандомизированный сигнал']
synonyms_en: ['noise', 'white noise', 'gaussian noise', 'noise generator', 'prng signal']
tags: ['GPU', 'PRNG', 'noise generation', 'signal processing']
---

# `signal_gen::NoiseGenerator` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo signal_generators --class NoiseGenerator`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=signal_generators__noise_generator__class_overview__v1 -->

**ЧТО**: Генерирует белый шум с использованием rocRAND или Philox PRNG на CPU/GPU

**ЗАЧЕМ**: Предоставляет статистически однородный шум для тестирования и обработки сигналов в радарных системах

**КАК**: Использует inline PRNG для CPU, HIP-кэширование для GPU. Поддерживает batch-генерацию с задержкой измерений (prof_events).

**Пример**:
```cpp
#include "signal_generators/generators/noise_generator.hpp"
using namespace signal_gen;

int main() {
  NoiseGenerator gen;
  auto gpu_mem = gen.GenerateToGpu(SystemSampling{1e6, 1e-6}, 8);
  std::complex<float> cpu_buf[1024];
  gen.GenerateToCpu(SystemSampling{1e6, 1e-6}, cpu_buf, 1024);
  return 0;
}
```

<!-- /rag-block -->

## Связанные секции из Doc/

- `signal_generators__gpu__c2_container_002__v1` (c2_container): ``` ┌─────────────────────────────────────────────────────────────┐ │  signal_generators module                                   │ │                                                             │ │  ┌…
- `signal_generators__architecture__section_002__v1` (section): ``` ┌─────────────────────────────────────────────────────────────┐ │                     Signal Generators                        │ ├─────────────────────────────────────────────────────────────┤ │  …
- `signal_generators__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:signal_generators source:signal_generators/CLAUDE.md -->  # signal_generators — Repository Card  _Источник: `signal_generators/CLAUDE.md`_  # 🤖 CLAUDE — `signal_generators` …
- `signal_generators__gpu__c3_component__v1` (c3_component): ### C3 — Component  ``` signal_gen namespace │ ├── ISignalGenerator (interface) │   ├── + GenerateToCpu(SystemSampling, out*, size) │   ├── + GenerateToGpu(SystemSampling, beam_count) → cl_mem │   └──…
- `signal_generators__gpu__section_002__v1` (section): ### Какой класс выбрать  | Класс | Для чего | Выход | Задержка | |-------|----------|-------|----------| | `FormSignalGenerator` | CW/Chirp + шум, N антенн | `InputData<cl_mem>` | FIXED/LINEAR/RANDOM …

## Public-методы (4)

## Method 1: `GenerateToCpu`

**Сигнатура** (`noise_generator.hpp:76`):
```cpp
void GenerateToCpu(const SystemSampling& system, std::complex<float>* out, size_t out_size) override
```

**Параметры**:
- `system` — `const SystemSampling&`
- `out` — `std::complex<float>*` *(pointer)*
- `out_size` — `size_t`

**Doxygen-источник**:
```cpp
/**
     * @brief CPU reference генерация шума (white или Gaussian по NoiseParams::type).
     *
     * @param system Параметры дискретизации (fs, length).
     * @param out Выходной буфер [out_size] complex<float>.
     * @param out_size Размер буфера (должен быть >= system.length).
     */
```

## Method 2: `GenerateToGpu`

**Сигнатура** (`noise_generator.hpp:89`):
```cpp
cl_mem GenerateToGpu(const SystemSampling& system, size_t beam_count = 1) override
```

**Параметры**:
- `system` — `const SystemSampling&`
- `beam_count` — `size_t`

**Возвращает**: `cl_mem`

**Doxygen-источник**:
```cpp
/**
     * @brief GPU production: Philox+BoxMuller kernel, multi-beam.
     *
     * @param system Параметры дискретизации (fs, length).
     * @param beam_count Количество лучей в выходе.
     *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
     *
     * @return cl_mem [beam_count × system.length × complex<float>]; caller обязан clReleaseMemObject.
     *   @test_check result != nullptr
     */
```

## Method 3: `GenerateToGpu`

**Сигнатура** (`noise_generator.hpp:104`):
```cpp
cl_mem GenerateToGpu(const SystemSampling& system, size_t beam_count, ProfEvents* prof_events)
```

**Параметры**:
- `system` — `const SystemSampling&`
- `beam_count` — `size_t`
- `prof_events` — `ProfEvents*` *(pointer)*

**Возвращает**: `cl_mem`

**Doxygen-источник**:
```cpp
/**
     * @brief Генерация на GPU с опциональным сбором событий профилирования.
     * @param prof_events nullptr → production (zero overhead); &vec → benchmark
     *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
     *
     * Собирает события: "Kernel" (noise_kernel / Philox+BoxMuller)
     * @param system Параметры дискретизации (fs, length).
     * @param beam_count Количество лучей в выходе.
     *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
     * @return cl_mem [beam_count × system.length × complex<float>]; caller обязан clReleaseMemObject.
     *   @test_check result != nullptr
     */
```

## Method 4: `Kind`

**Сигнатура** (`noise_generator.hpp:114`):
```cpp
SignalKind Kind() const override { return SignalKind::NOISE;
```

**Возвращает**: `SignalKind`

**Doxygen-источник**:
```cpp
/**
     * @brief Возвращает тип сигнала (для introspection).
     *
     * @return Всегда `SignalKind::NOISE` для этого класса.
     *   @test_check result == SignalKind::NOISE
     */
```

