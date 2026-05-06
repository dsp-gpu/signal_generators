<!-- type:meta_cmake_specific repo:signal_generators inherits:dsp_gpu__root__meta_cmake_common__v1 -->

# CMake Specific — signal_generators

```yaml
inherits: dsp_gpu__root__meta_cmake_common__v1
specific_only: true
target: DspSignalGenerators
description: "Signal generators: CW, LFM, Noise, Script, FormSignal"
adds_find_package: [hip, hiprtc]
adds_links: [DspCore::DspCore, DspSpectrum::DspSpectrum]
```

## Project

- **Target**: `DspSignalGenerators`
- **Описание**: Signal generators: CW, LFM, Noise, Script, FormSignal

## Уникальные find_package

```cmake
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)
```

## Линкуемые библиотеки

```cmake
target_link_libraries(DspSignalGenerators PUBLIC
  DspCore::DspCore
  DspSpectrum::DspSpectrum
)
```

## Исходники (10 файлов)

```cmake
target_sources(DspSignalGenerators PRIVATE
  src/signal_generators/src/cw_generator_rocm.cpp
  src/signal_generators/src/lfm_generator_rocm.cpp
  src/signal_generators/src/lfm_conjugate_generator_rocm.cpp
  src/signal_generators/src/lfm_generator_analytical_delay_rocm.cpp
  src/signal_generators/src/noise_generator_rocm.cpp
  src/signal_generators/src/script_generator_rocm.cpp
  src/signal_generators/src/form_signal_generator_rocm.cpp
  src/signal_generators/src/delayed_form_signal_generator_rocm.cpp
  src/signal_generators/src/form_script_generator_rocm.cpp
  src/signal_generators/src/signal_generator_factory.cpp
)
```

## Прочие специфичные строки (22)

```cmake
DESCRIPTION "Signal generators: CW, LFM, Noise, Script, FormSignal"
NAMES hiprtc
PATH_SUFFIXES lib lib64
PRIVATE ${HIPRTC_LIB}
PUBLIC  <TARGET>::<TARGET> <TARGET>::<TARGET>
REQUIRED
fetch_dsp_spectrum()
find_library(HIPRTC_LIB
find_package(hip   REQUIRED)
find_package(hiprtc REQUIRED)
message(STATUS "[<TARGET>] hiprtc: ${HIPRTC_LIB}")
src/signal_generators/src/cw_generator_rocm.cpp
src/signal_generators/src/delayed_form_signal_generator_rocm.cpp
src/signal_generators/src/form_script_generator_rocm.cpp
src/signal_generators/src/form_signal_generator_rocm.cpp
src/signal_generators/src/lfm_conjugate_generator_rocm.cpp
src/signal_generators/src/lfm_generator_analytical_delay_rocm.cpp
src/signal_generators/src/lfm_generator_rocm.cpp
src/signal_generators/src/noise_generator_rocm.cpp
src/signal_generators/src/script_generator_rocm.cpp
src/signal_generators/src/signal_generator_factory.cpp
target_link_libraries(<TARGET>
```

