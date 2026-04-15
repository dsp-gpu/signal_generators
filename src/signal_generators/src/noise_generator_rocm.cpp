/**
 * @file noise_generator_rocm.cpp
 * @brief NoiseGeneratorROCm implementation — Noise on GPU (ROCm/HIP)
 *
 * Philox-2x32-10 PRNG + Box-Muller (Gaussian) or uniform (White).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include <signal_generators/generators/noise_generator_rocm.hpp>
#include <signal_generators/kernels/noise_kernels_rocm.hpp>
#include <spectrum/utils/rocm_profiling_helpers.hpp>
#include <core/services/scoped_hip_event.hpp>
#include <core/services/console_output.hpp>

#include <stdexcept>
#include <cmath>

using fft_func_utils::MakeROCmDataFromEvents;
using drv_gpu_lib::ScopedHipEvent;

namespace signal_gen {

static const std::vector<std::string> kNoiseKernelNames = {
  "generate_noise_gaussian", "generate_noise_white"
};

NoiseGeneratorROCm::NoiseGeneratorROCm(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "NoiseGen", "modules/signal_generators/kernels") {
}

void NoiseGeneratorROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetNoiseSource_rocm(), kNoiseKernelNames);
  compiled_ = true;
}

drv_gpu_lib::InputData<void*> NoiseGeneratorROCm::GenerateToGpu(
    const SystemSampling& system,
    const NoiseParams& params,
    uint32_t beam_count,
    ROCmProfEvents* prof_events) {

  EnsureCompiled();

  size_t total = static_cast<size_t>(beam_count) * system.length;
  size_t buffer_size = total * sizeof(std::complex<float>);

  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error("NoiseGeneratorROCm: hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  // Select kernel by noise type
  const char* kernel_name = (params.type == NoiseType::GAUSSIAN)
      ? "generate_noise_gaussian"
      : "generate_noise_white";
  hipFunction_t k = ctx_.GetKernel(kernel_name);

  // Seed: use provided or generate random
  uint32_t seed = (params.seed == 0)
      ? static_cast<uint32_t>(rng_())
      : static_cast<uint32_t>(params.seed);

  unsigned int total_u = static_cast<unsigned int>(total);
  float power_param = static_cast<float>(
      (params.type == NoiseType::GAUSSIAN) ? std::sqrt(params.power) : std::sqrt(params.power));

  void* args[] = { &output_ptr, &total_u, &power_param, &seed };

  // 1D grid (all samples in one dimension)
  unsigned int grid = (total_u + kBlockSize - 1) / kBlockSize;

  ScopedHipEvent ev_s, ev_e;
  if (prof_events) {
    ev_s.Create(); ev_e.Create();
    hipEventRecord(ev_s.get(), ctx_.stream());
  }

  err = hipModuleLaunchKernel(k,
      grid, 1, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess) {
    (void)hipFree(output_ptr);
    throw std::runtime_error("NoiseGeneratorROCm: kernel launch failed: " +
                              std::string(hipGetErrorString(err)));
  }

  if (prof_events) hipEventRecord(ev_e.get(), ctx_.stream());
  hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"Kernel",
        MakeROCmDataFromEvents(ev_s.get(), ev_e.get(), 0, kernel_name)});
  }

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = beam_count;
  result.n_point = static_cast<uint32_t>(system.length);
  result.data = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

std::vector<std::complex<float>> NoiseGeneratorROCm::GenerateToCpu(
    const SystemSampling& system,
    const NoiseParams& params,
    uint32_t beam_count) {

  size_t total = static_cast<size_t>(beam_count) * system.length;
  std::vector<std::complex<float>> output(total);

  std::mt19937 gen(params.seed == 0 ? std::random_device{}() : static_cast<unsigned>(params.seed));

  if (params.type == NoiseType::GAUSSIAN) {
    float std_dev = static_cast<float>(std::sqrt(params.power));
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (size_t i = 0; i < total; ++i) {
      output[i] = std::complex<float>(dist(gen), dist(gen));
    }
  } else {
    float amp = static_cast<float>(std::sqrt(params.power));
    std::uniform_real_distribution<float> dist(-amp, amp);
    for (size_t i = 0; i < total; ++i) {
      output[i] = std::complex<float>(dist(gen), dist(gen));
    }
  }
  return output;
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
