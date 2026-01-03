#include <metal_stdlib>
using namespace metal;

struct AdamWParams {
    uint n;
    float step_size;
    float beta1;
    float beta2;
    float eps;
    float lr_wd; // lr * weight_decay
};

kernel void adamw_master_step_fp16(
    device half* p                 [[ buffer(0) ]], // fp16 params (updated in-place)
    device const half* g           [[ buffer(1) ]], // fp16 grads
    device float* master           [[ buffer(2) ]], // fp32 master weights (updated)
    device float* exp_avg          [[ buffer(3) ]], // fp32 m (updated)
    device float* exp_avg_sq       [[ buffer(4) ]], // fp32 v (updated)
    constant AdamWParams& prm      [[ buffer(5) ]],
    uint tid                       [[ thread_position_in_threadgroup ]],
    uint tg_id                     [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    const uint i = tg_id * TG + tid;
    if (i >= prm.n) {
        return;
    }

    const float grad = float(g[i]);
    float w = master[i];

    if (prm.lr_wd != 0.0f) {
        w = w * (1.0f - prm.lr_wd);
    }

    float m = exp_avg[i];
    float v = exp_avg_sq[i];

    m = prm.beta1 * m + (1.0f - prm.beta1) * grad;
    v = prm.beta2 * v + (1.0f - prm.beta2) * (grad * grad);

    const float denom = sqrt(v) + prm.eps;
    w = w - prm.step_size * (m / denom);

    exp_avg[i] = m;
    exp_avg_sq[i] = v;
    master[i] = w;
    p[i] = half(w);
}

