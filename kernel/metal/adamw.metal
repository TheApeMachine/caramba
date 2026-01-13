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

template <typename T>
inline void adamw_master_step_impl(
    device T* p,
    device const T* g,
    device float* master,
    device float* exp_avg,
    device float* exp_avg_sq,
    constant AdamWParams& prm,
    uint tid,
    uint tg_id
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
    p[i] = T(w);
}

kernel void adamw_master_step_fp16(
    device half* p                 [[ buffer(0) ]],
    device const half* g           [[ buffer(1) ]],
    device float* master           [[ buffer(2) ]],
    device float* exp_avg          [[ buffer(3) ]],
    device float* exp_avg_sq       [[ buffer(4) ]],
    constant AdamWParams& prm      [[ buffer(5) ]],
    uint tid                       [[ thread_position_in_threadgroup ]],
    uint tg_id                     [[ threadgroup_position_in_grid ]]
) {
    adamw_master_step_impl<half>(p, g, master, exp_avg, exp_avg_sq, prm, tid, tg_id);
}

kernel void adamw_master_step_fp32(
    device float* p                [[ buffer(0) ]],
    device const float* g          [[ buffer(1) ]],
    device float* master           [[ buffer(2) ]],
    device float* exp_avg          [[ buffer(3) ]],
    device float* exp_avg_sq       [[ buffer(4) ]],
    constant AdamWParams& prm      [[ buffer(5) ]],
    uint tid                       [[ thread_position_in_threadgroup ]],
    uint tg_id                     [[ threadgroup_position_in_grid ]]
) {
    adamw_master_step_impl<float>(p, g, master, exp_avg, exp_avg_sq, prm, tid, tg_id);
}
