/**
 * @file dsp_signal_generators_module.cpp
 * @brief pybind11 bindings for dsp::signal_generators
 *
 * Python API:
 *   import dsp_signal_generators as sg
 *   gen = sg.FormSignalGeneratorROCm(ctx)
 *
 * Экспортируемые классы:
 *   FormSignalGeneratorROCm        — генератор формы сигнала (ROCm)
 *   DelayedFormSignalGeneratorROCm — с задержкой (ROCm)
 *   LfmAnalyticalDelayGenerator    — ЛЧМ аналитическая задержка
 *   LfmAnalyticalDelayGeneratorROCm
 *
 * Note: CW/LFM/Noise/Script генераторы — в базовом gpuworklib (OpenCL),
 *   они используют signal_service.hpp который остался в GPUWorkLib.
 *   Чистые ROCm-версии добавляются здесь.
 */

#include "py_helpers.hpp"

#if ENABLE_ROCM
#include "py_gpu_context.hpp"
#include "py_form_signal_rocm.hpp"
#include "py_delayed_form_signal_rocm.hpp"
#include "py_lfm_analytical_delay_rocm.hpp"
#endif

// py_lfm_analytical_delay.hpp — OpenCL GPUContext, только nvidia-ветка
// #include "py_lfm_analytical_delay.hpp"

PYBIND11_MODULE(dsp_signal_generators, m) {
    m.doc() = "dsp::signal_generators — ROCm signal generators\n\n"
              "Classes:\n"
              "  ROCmGPUContext                  - GPU context (AMD ROCm)\n"
              "  FormSignalGeneratorROCm        - form signal generator (ROCm)\n"
              "  DelayedFormSignalGeneratorROCm - with fractional delay (ROCm)\n"
              "  LfmAnalyticalDelayGenerator    - LFM analytical delay\n";

    // register_lfm_analytical_delay(m);  // OpenCL-only, nvidia-ветка

#if ENABLE_ROCM
    py::class_<ROCmGPUContext>(m, "ROCmGPUContext",
        "ROCm GPU context (creates HIP backend for AMD GPU).")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def_property_readonly("device_name", &ROCmGPUContext::device_name)
        .def_property_readonly("device_index", &ROCmGPUContext::device_index);

    register_form_signal_rocm(m);
    register_delayed_form_signal_rocm(m);
    register_lfm_analytical_delay_rocm(m);
#endif
}
