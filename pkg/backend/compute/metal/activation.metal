// Compile with:
// xcrun -sdk macosx metal -c activation.metal -o activation.air && xcrun -sdk macosx metallib activation.air -o activation.metallib

#include <metal_stdlib>
using namespace metal;

kernel void relu_forward(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    dst[i] = max(src[i], 0.0f);
}

kernel void leaky_relu_forward(
    device const float* src  [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant float& alpha    [[buffer(2)]],
    uint i                   [[thread_position_in_grid]])
{
    float x = src[i];
    dst[i] = x >= 0.0f ? x : alpha * x;
}

kernel void gelu_forward(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    float x = src[i];
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    float bounded = clamp(inner, -20.0f, 20.0f);
    dst[i] = 0.5f * x * (1.0f + tanh(bounded));
}

kernel void tanh_forward(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    dst[i] = tanh(src[i]);
}

kernel void sigmoid_forward(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    dst[i] = 1.0f / (1.0f + exp(-src[i]));
}

kernel void swish_forward(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    float x = src[i];
    float sig = 1.0f / (1.0f + exp(-x));
    dst[i] = x * sig;
}

kernel void selu_forward(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    constexpr float alpha = 1.6732632423543772f;
    constexpr float scale = 1.0507009873554805f;
    float x = src[i];
    dst[i] = x > 0.0f ? scale * x : scale * alpha * (exp(x) - 1.0f);
}

kernel void swiglu_forward(
    device const float* src  [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint i                   [[thread_position_in_grid]])
{
    if (i >= n) return;
    float gate  = src[i];
    float value = src[n + i];
    float sig   = 1.0f / (1.0f + exp(-gate));
    dst[i] = gate * sig * value;
}
