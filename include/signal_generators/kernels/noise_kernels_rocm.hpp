#pragma once

/**
 * @brief HIP kernel-source для Noise-генератора (Philox + Box-Muller / uniform).
 *
 * @note Тип B (technical header): R"HIP(...)HIP" source для hiprtc.
 *       Kernels:
 *         - generate_noise_gaussian — комплексный Gaussian шум, std_dev задаётся
 *                                     (Philox-2x32-10 → Box-Muller → re=r·cos, im=r·sin)
 *         - generate_noise_white    — равномерный шум на [-amplitude, +amplitude]
 *       PRNG: Philox-2x32-10 (counter-based) — воспроизводим при одинаковом seed.
 *       Используется __umulhi (HIP intrinsic) — быстрее чем 64-bit mul на RDNA4.
 *       1D grid: total_points / blockDim.x. __launch_bounds__(256).
 * @note Порт из noise_kernel.cl + prng.cl (OpenCL → HIP/ROCm).
 *
 * История:
 *   - Создан:  2026-03-14
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

namespace signal_gen {
namespace kernels {

inline const char* GetNoiseSource_rocm() {
  return R"HIP(

struct float2_t {
    float x;
    float y;
};

struct uint2_t {
    unsigned int x;
    unsigned int y;
};

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ════════════════════════════════════════════════════════════════════════════
// Philox-2x32-10 PRNG (counter-based)
// ════════════════════════════════════════════════════════════════════════════

__device__ uint2_t philox2x32_round(uint2_t ctr, unsigned int key) {
    const unsigned int PHILOX_M = 0xD2511F53u;
    unsigned int lo = ctr.x * PHILOX_M;
    unsigned int hi = __umulhi(ctr.x, PHILOX_M);
    uint2_t result;
    result.x = hi ^ key ^ ctr.y;
    result.y = lo;
    return result;
}

__device__ uint2_t philox2x32_10(uint2_t ctr, unsigned int key) {
    const unsigned int PHILOX_BUMP = 0x9E3779B9u;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key);
    return ctr;
}

// ════════════════════════════════════════════════════════════════════════════
// Noise kernels
// ════════════════════════════════════════════════════════════════════════════

extern "C" __global__ __launch_bounds__(256)
void generate_noise_gaussian(
    float2_t* __restrict__ output,
    const unsigned int total_points,
    const float std_dev,
    const unsigned int seed)
{
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_points) return;

    uint2_t ctr;
    ctr.x = gid;
    ctr.y = seed;
    uint2_t rnd = philox2x32_10(ctr, 0xCD9E8D57u);

    float u1 = (float)(rnd.x) / 4294967296.0f + 1e-10f;
    float u2 = (float)(rnd.y) / 4294967296.0f;

    float r = __fsqrt_rn(-2.0f * __logf(u1)) * std_dev;
    float cos_val, sin_val;
    __sincosf(2.0f * M_PI_F * u2, &sin_val, &cos_val);

    float2_t out;
    out.x = r * cos_val;
    out.y = r * sin_val;
    output[gid] = out;
}

extern "C" __global__ __launch_bounds__(256)
void generate_noise_white(
    float2_t* __restrict__ output,
    const unsigned int total_points,
    const float amplitude,
    const unsigned int seed)
{
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_points) return;

    uint2_t ctr;
    ctr.x = gid;
    ctr.y = seed;
    uint2_t rnd = philox2x32_10(ctr, 0xCD9E8D57u);

    float re = ((float)(rnd.x) / 4294967296.0f * 2.0f - 1.0f) * amplitude;
    float im = ((float)(rnd.y) / 4294967296.0f * 2.0f - 1.0f) * amplitude;

    float2_t out;
    out.x = re;
    out.y = im;
    output[gid] = out;
}

)HIP";
}

}  // namespace kernels
}  // namespace signal_gen
