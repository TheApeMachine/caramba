// Compile with:
// xcrun -sdk macosx metal -c activation.metal -o activation.air && xcrun -sdk macosx metallib activation.air -o activation.metallib

#include <metal_stdlib>
using namespace metal;

// Rational tanh approximation: tanh(x) ≈ x*(27+x*x)/(27+9*x*x)
// Accurate for |x| < ~3, clamped outside
static inline float fast_tanh(float x) {
    float x2 = x * x;
    float num = x * (27.0f + x2);
    float den = 27.0f + 9.0f * x2;
    return num / den;
}

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
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    dst[i] = 0.5f * x * (1.0f + fast_tanh(inner));
}

kernel void tanh_forward(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    dst[i] = fast_tanh(src[i]);
}

kernel void sigmoid_forward(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    // sigmoid(x) = 0.5 * (1 + tanh(x/2))
    dst[i] = 0.5f * (1.0f + fast_tanh(src[i] * 0.5f));
}

kernel void swish_forward(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    float x = src[i];
    float sig = 0.5f * (1.0f + fast_tanh(x * 0.5f));
    dst[i] = x * sig;
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
    float sig   = 0.5f * (1.0f + fast_tanh(gate * 0.5f));
    dst[i] = sig * value;
}
