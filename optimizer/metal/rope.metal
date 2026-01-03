#include <metal_stdlib>
using namespace metal;

// Must match `RoPEParams` in `ops.mm` (layout + types).
struct RoPEParams {
    uint d_model;
    uint rot_dim;
    uint half_rot;
    uint seq_len;
};

// Apply RoPE in the same "half split" layout as `layer/rope.py`:
// x1 = x[..., :rot/2], x2 = x[..., rot/2:rot]
// y1 = x1*cos - x2*sin
// y2 = x1*sin + x2*cos
kernel void rope_fp16(
    device const half* x     [[ buffer(0) ]], // (B*H*T, D)
    device const half* cos_t [[ buffer(1) ]], // (T, rot/2)
    device const half* sin_t [[ buffer(2) ]], // (T, rot/2)
    device half* out         [[ buffer(3) ]], // (B*H*T, D)
    constant RoPEParams& p   [[ buffer(4) ]],
    uint tid                 [[ thread_position_in_threadgroup ]],
    uint tg_id               [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;

    const uint vec = tg_id;
    const uint t = (p.seq_len > 0) ? (vec % p.seq_len) : 0;

    device const half* xr = x + vec * p.d_model;
    device half* yr = out + vec * p.d_model;

    device const half* c = cos_t + t * p.half_rot;
    device const half* s = sin_t + t * p.half_rot;

    for (uint i = tid; i < p.d_model; i += TG) {
        if (i < p.half_rot) {
            const float x1 = float(xr[i]);
            const float x2 = float(xr[i + p.half_rot]);
            const float cc = float(c[i]);
            const float ss = float(s[i]);
            yr[i] = half(x1 * cc - x2 * ss);
            yr[i + p.half_rot] = half(x1 * ss + x2 * cc);
        } else if (i >= p.rot_dim) {
            yr[i] = xr[i];
        }
        // i in [half_rot, rot_dim) is written by the corresponding i-half_rot thread.
    }
}

kernel void rope_bwd_fp16(
    device const half* grad_y [[ buffer(0) ]], // (B*H*T, D)
    device const half* cos_t  [[ buffer(1) ]], // (T, rot/2)
    device const half* sin_t  [[ buffer(2) ]], // (T, rot/2)
    device half* grad_x       [[ buffer(3) ]], // (B*H*T, D)
    constant RoPEParams& p    [[ buffer(4) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;

    const uint vec = tg_id;
    const uint t = (p.seq_len > 0) ? (vec % p.seq_len) : 0;

    device const half* gr = grad_y + vec * p.d_model;
    device half* gx = grad_x + vec * p.d_model;

    device const half* c = cos_t + t * p.half_rot;
    device const half* s = sin_t + t * p.half_rot;

    for (uint i = tid; i < p.d_model; i += TG) {
        if (i < p.half_rot) {
            // y1 = x1*c - x2*s
            // y2 = x1*s + x2*c
            // => x1 = y1*c + y2*s
            //    x2 = -y1*s + y2*c
            const float gy1 = float(gr[i]);
            const float gy2 = float(gr[i + p.half_rot]);
            const float cc = float(c[i]);
            const float ss = float(s[i]);
            gx[i] = half(gy1 * cc + gy2 * ss);
            gx[i + p.half_rot] = half(-gy1 * ss + gy2 * cc);
        } else if (i >= p.rot_dim) {
            gx[i] = gr[i];
        }
        // i in [half_rot, rot_dim) is written by the corresponding i-half_rot thread.
    }
}

