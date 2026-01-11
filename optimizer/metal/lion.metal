#include <metal_stdlib>
using namespace metal;

// Must match `LionParams` in `ops.mm`.
struct LionParams {
    uint n;
    float lr;
    float beta1;
    float weight_decay;
};

template <typename T>
inline void lion_step_impl(
    device T* p,
    device const T* g,
    device T* m,
    constant LionParams& prm,
    uint tid,
    uint tg_id
) {
    constexpr uint TG = 256;
    const uint i = tg_id * TG + tid;
    if (i >= prm.n) {
        return;
    }

    const float grad = float(g[i]);
    const float m0 = float(m[i]);
    const float m1 = prm.beta1 * m0 + (1.0f - prm.beta1) * grad;

    float p0 = float(p[i]);
    if (prm.weight_decay != 0.0f) {
        p0 = p0 * (1.0f - prm.lr * prm.weight_decay);
    }
    const float s = (m1 > 0.0f) ? 1.0f : ((m1 < 0.0f) ? -1.0f : 0.0f);
    const float p1 = p0 - prm.lr * s;

    p[i] = T(p1);
    m[i] = T(m1);
}

kernel void lion_step_fp16(
    device half* p           [[ buffer(0) ]],
    device const half* g     [[ buffer(1) ]],
    device half* m           [[ buffer(2) ]],
    constant LionParams& prm [[ buffer(3) ]],
    uint tid                 [[ thread_position_in_threadgroup ]],
    uint tg_id               [[ threadgroup_position_in_grid ]]
) {
    lion_step_impl<half>(p, g, m, prm, tid, tg_id);
}

kernel void lion_step_fp32(
    device float* p          [[ buffer(0) ]],
    device const float* g    [[ buffer(1) ]],
    device float* m          [[ buffer(2) ]],
    constant LionParams& prm [[ buffer(3) ]],
    uint tid                 [[ thread_position_in_threadgroup ]],
    uint tg_id               [[ threadgroup_position_in_grid ]]
) {
    lion_step_impl<float>(p, g, m, prm, tid, tg_id);
}
