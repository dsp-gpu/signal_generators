/**
 * @file lfm_generator_rocm.cpp
 * @brief LfmGeneratorROCm implementation — LFM chirp on GPU (ROCm/HIP)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include <signal_generators/generators/lfm_generator_rocm.hpp>
#include <signal_generators/kernels/lfm_kernels_rocm.hpp>
#include <spectrum/utils/rocm_profiling_helpers.hpp>
#include <core/services/scoped_hip_event.hpp>
#include <core/services/console_output.hpp>

#include <stdexcept>
#include <cmath>

using fft_func_utils::MakeROCmDataFromEvents;
using drv_gpu_lib::ScopedHipEvent;

namespace signal_gen {

static const std::vector<std::string> kLfmKernelNames = {
  "generate_lfm", "generate_lfm_real"
};

LfmGeneratorROCm::LfmGeneratorROCm(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "LfmGen", "modules/signal_generators/kernels") {
}

void LfmGeneratorROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetLfmSource_rocm(), kLfmKernelNames);
  compiled_ = true;
}

drv_gpu_lib::InputData<void*> LfmGeneratorROCm::GenerateToGpu(
    const SystemSampling& system,
    const LfmParams& params,
    uint32_t beam_count,
    ROCmProfEvents* prof_events) {

  EnsureCompiled();

  size_t total = static_cast<size_t>(beam_count) * system.length;
  size_t buffer_size = total * sizeof(std::complex<float>);

  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error("LfmGeneratorROCm: hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  const char* kernel_name = params.complex_iq ? "generate_lfm" : "generate_lfm_real";
  hipFunction_t k = ctx_.GetKernel(kernel_name);

  double duration = static_cast<double>(system.length) / system.fs;
  float chirp_rate = static_cast<float>(params.GetChirpRate(duration));

  unsigned int bc = beam_count;
  unsigned int np = static_cast<unsigned int>(system.length);
  float fs   = static_cast<float>(system.fs);
  float f_st = static_cast<float>(params.f_start);
  float amp  = static_cast<float>(params.amplitude);

  void* args[] = { &output_ptr, &bc, &np, &fs, &f_st, &chirp_rate, &amp };

  unsigned int grid_x = (np + kBlockSize - 1) / kBlockSize;

  ScopedHipEvent ev_s, ev_e;
  if (prof_events) {
    ev_s.Create(); ev_e.Create();
    hipEventRecord(ev_s.get(), ctx_.stream());
  }

  err = hipModuleLaunchKernel(k,
      grid_x, bc, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess) {
    (void)hipFree(output_ptr);
    throw std::runtime_error("LfmGeneratorROCm: kernel launch failed: " +
                              std::string(hipGetErrorString(err)));
  }

  if (prof_events) hipEventRecord(ev_e.get(), ctx_.stream());
  hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"Kernel",
        MakeROCmDataFromEvents(ev_s.get(), ev_e.get(), 0, "generate_lfm")});
  }

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = beam_count;
  result.n_point = static_cast<uint32_t>(system.length);
  result.data = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

std::vector<std::complex<float>> LfmGeneratorROCm::GenerateToCpu(
    const SystemSampling& system,
    const LfmParams& params,
    uint32_t beam_count) {

  size_t total = static_cast<size_t>(beam_count) * system.length;
  std::vector<std::complex<float>> output(total);
  double duration = static_cast<double>(system.length) / system.fs;
  double k = params.GetChirpRate(duration);

  for (uint32_t b = 0; b < beam_count; ++b) {
    for (size_t n = 0; n < system.length; ++n) {
      double t = static_cast<double>(n) / system.fs;
      double phase = M_PI * k * t * t + 2.0 * M_PI * params.f_start * t;
      size_t idx = b * system.length + n;
      if (params.complex_iq) {
        output[idx] = std::complex<float>(
            static_cast<float>(params.amplitude * std::cos(phase)),
            static_cast<float>(params.amplitude * std::sin(phase)));
      } else {
        output[idx] = std::complex<float>(
            static_cast<float>(params.amplitude * std::cos(phase)), 0.0f);
      }
    }
  }
  return output;
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
