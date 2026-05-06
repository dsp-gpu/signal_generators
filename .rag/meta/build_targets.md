<!-- type:meta_targets repo:signal_generators source:signal_generators/CMakeLists.txt -->

# Build Targets — signal_generators

## Targets

- **`DspSignalGenerators`** (library)
  - PUBLIC: `DspCore::DspCore`, `DspSpectrum::DspSpectrum`
  - PRIVATE: `${HIPRTC_LIB}`

## BUILD-флаги (option)

- `DSP_SIGNAL_GENERATORS_BUILD_TESTS` (default `ON`) — Build tests
- `DSP_SIGNAL_GENERATORS_BUILD_PYTHON` (default `OFF`) — Build Python bindings

## Зависимости от DSP репо

- `core` — через `fetch_dsp_core()`
- `spectrum` — через `fetch_dsp_spectrum()`

## External find_package

- `hip` (required)
- `hiprtc` (required)
