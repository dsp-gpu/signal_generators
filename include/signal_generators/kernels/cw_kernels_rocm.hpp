#pragma once

/**
 * @brief HIP kernel-source для CW-генератора (continuous wave).
 *
 * @note Тип B (technical header): R"HIP(...)HIP" source для hiprtc.
 *       Kernels:
 *         - generate_cw       — комплексный выход s(t) = A·exp(j·(2π·f·t + φ₀))
 *         - generate_cw_real  — действительный выход s(t) = A·cos(2π·f·t + φ₀), Im=0
 *       Multi-beam: freq_i = base_freq + beam_id · freq_step.
 *       2D grid: (n_point ÷ blockDim.x, beam_count). __launch_bounds__(256).
 *       Используется __sincosf / __cosf — fast-math, погрешность ~2⁻²³.
 * @note Порт из cw_kernel.cl (OpenCL → HIP/ROCm).
 *
 * История:
 *   - Создан:  2026-03-14
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

namespace signal_gen {
namespace kernels {

inline const char* GetCwSource_rocm() {
  return R"HIP(

struct float2_t {
    float x;
    float y;
};

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

extern "C" __global__ __launch_bounds__(256)
void generate_cw(
    float2_t* __restrict__ output,
    const unsigned int beam_count,
    const unsigned int n_point,
    const float sample_rate,
    const float base_freq,
    const float freq_step,
    const float amplitude,
    const float initial_phase)
{
    const unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int beam_id   = blockIdx.y;
    if (sample_id >= n_point || beam_id >= beam_count) return;

    const unsigned int gid = beam_id * n_point + sample_id;

    const float freq = base_freq + (float)beam_id * freq_step;
    const float t = (float)sample_id / sample_rate;
    const float phase = 2.0f * M_PI_F * freq * t + initial_phase;

    float cos_val, sin_val;
    __sincosf(phase, &sin_val, &cos_val);

    float2_t out;
    out.x = amplitude * cos_val;
    out.y = amplitude * sin_val;
    output[gid] = out;
}

extern "C" __global__ __launch_bounds__(256)
void generate_cw_real(
    float2_t* __restrict__ output,
    const unsigned int beam_count,
    const unsigned int n_point,
    const float sample_rate,
    const float base_freq,
    const float freq_step,
    const float amplitude,
    const float initial_phase)
{
    const unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int beam_id   = blockIdx.y;
    if (sample_id >= n_point || beam_id >= beam_count) return;

    const unsigned int gid = beam_id * n_point + sample_id;

    const float freq = base_freq + (float)beam_id * freq_step;
    const float t = (float)sample_id / sample_rate;
    const float phase = 2.0f * M_PI_F * freq * t + initial_phase;

    float2_t out;
    out.x = amplitude * __cosf(phase);
    out.y = 0.0f;
    output[gid] = out;
}

)HIP";
}

}  // namespace kernels
}  // namespace signal_gen
