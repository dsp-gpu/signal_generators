cmake_minimum_required(VERSION 3.25)
project(DspSignalGenerators VERSION 0.1.0 LANGUAGES CXX HIP)

# ── Опции ──────────────────────────────────────────────────────────
option(DSP_SIGNAL_GENERATORS_BUILD_TESTS  "Build tests"            ON)
option(DSP_SIGNAL_GENERATORS_BUILD_PYTHON "Build Python bindings"  OFF)

# ── Зависимости ────────────────────────────────────────────────────
find_package(hip   REQUIRED)   # lowercase — Linux case-sensitive!
find_package(hiprtc REQUIRED)

include(cmake/fetch_deps.cmake)
fetch_dsp_core()
fetch_dsp_spectrum()

# ── Библиотека ─────────────────────────────────────────────────────
add_library(DspSignalGenerators STATIC)
add_library(DspSignalGenerators::DspSignalGenerators ALIAS DspSignalGenerators)

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

target_include_directories(DspSignalGenerators
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    src/signal_generators/include
    kernels/
    src
)
target_link_libraries(DspSignalGenerators
  PUBLIC  DspCore::DspCore DspSpectrum::DspSpectrum
  PRIVATE hiprtc
)
target_compile_features(DspSignalGenerators PUBLIC cxx_std_17)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS DspSignalGenerators EXPORT DspSignalGeneratorsTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/DspSignalGeneratorsConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/DspSignalGeneratorsConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspSignalGenerators
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DspSignalGeneratorsConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)
install(EXPORT DspSignalGeneratorsTargets NAMESPACE DspSignalGenerators::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspSignalGenerators
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/DspSignalGeneratorsConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/DspSignalGeneratorsConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspSignalGenerators
)

if(DSP_SIGNAL_GENERATORS_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(DSP_SIGNAL_GENERATORS_BUILD_PYTHON)
  add_subdirectory(python)
endif()
