#include <metal_stdlib>
using namespace metal;

struct ResonantUpdateParams {
  uint32_t n;
  uint32_t D;
  uint32_t H;
  float inv_D;
  float scale;
  float damping;
  uint32_t zero_diag;
};

kernel void resonant_update_fwd_fp32(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device const float* vr [[buffer(2)]],
    device const float* vi [[buffer(3)]],
    device const float* diag [[buffer(4)]],  // (H*D)
    device float* xo [[buffer(5)]],
    device float* yo [[buffer(6)]],
    device float* a_out [[buffer(7)]],
    device float* b_out [[buffer(8)]],
    device float* inv_r_out [[buffer(9)]],
    constant ResonantUpdateParams& p [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  const uint32_t d = gid % p.D;
  const uint32_t tmp = gid / p.D;
  const uint32_t h = tmp % p.H;
  const float diagv = diag[h * p.D + d];

  const float one_minus = 1.0f - p.damping;
  float cr = vr[gid] * p.inv_D;
  float ci = vi[gid] * p.inv_D;
  if (p.zero_diag) {
    cr -= diagv * x[gid];
    ci -= diagv * y[gid];
  }
  const float a = x[gid] * one_minus + p.scale * cr;
  const float b = y[gid] * one_minus + p.scale * ci;
  // Some Metal toolchains don’t expose `rsqrt` as an identifier here.
  // Use a portable form instead.
  const float inv_r = 1.0f / sqrt(a * a + b * b + 1e-6f);

  xo[gid] = a * inv_r;
  yo[gid] = b * inv_r;
  a_out[gid] = a;
  b_out[gid] = b;
  inv_r_out[gid] = inv_r;
}

kernel void resonant_update_bwd_fp32(
    device const float* gxo [[buffer(0)]],
    device const float* gyo [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device const float* y [[buffer(3)]],
    device const float* diag [[buffer(4)]],
    device const float* a [[buffer(5)]],
    device const float* b [[buffer(6)]],
    device const float* inv_r [[buffer(7)]],
    device float* gvr [[buffer(8)]],
    device float* gvi [[buffer(9)]],
    device float* gx [[buffer(10)]],
    device float* gy [[buffer(11)]],
    constant ResonantUpdateParams& p [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  const uint32_t d = gid % p.D;
  const uint32_t tmp = gid / p.D;
  const uint32_t h = tmp % p.H;
  const float diagv = diag[h * p.D + d];

  const float invr = inv_r[gid];
  const float invr2 = invr * invr;
  const float invr3 = invr2 * invr;
  const float dot = gxo[gid] * a[gid] + gyo[gid] * b[gid];
  const float ga = gxo[gid] * invr - a[gid] * dot * invr3;
  const float gb = gyo[gid] * invr - b[gid] * dot * invr3;

  float coeff = 1.0f - p.damping;
  if (p.zero_diag) {
    coeff -= p.scale * diagv;
  }
  gx[gid] = ga * coeff;
  gy[gid] = gb * coeff;
  gvr[gid] = ga * (p.scale * p.inv_D);
  gvi[gid] = gb * (p.scale * p.inv_D);
}

