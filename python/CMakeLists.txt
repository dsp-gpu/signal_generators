# ── DspSignalGenerators Python bindings ───────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(dsp_signal_generators dsp_signal_generators_module.cpp)
target_link_libraries(dsp_signal_generators PRIVATE
  DspSignalGenerators::DspSignalGenerators)
target_include_directories(dsp_signal_generators PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/signal_generators/include
  # py_gpu_context.hpp — ROCmGPUContext shared header
  ${FETCHCONTENT_SOURCE_DIR_DSPCORE}/python)
target_compile_definitions(dsp_signal_generators PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>)
install(TARGETS dsp_signal_generators DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
