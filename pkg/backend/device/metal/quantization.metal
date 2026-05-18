#include <metal_stdlib>

using namespace metal;

kernel void int8_dequant(
    device const char* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    out[index] = float(input[index]);
}

kernel void int4_dequant(
    device const uchar* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    uchar packed = input[index >> 1u];
    int value = int((index & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u));

    if (value >= 8) {
        value -= 16;
    }

    out[index] = float(value);
}

kernel void int8_quant(
    device const float* input [[buffer(0)]],
    device char* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    float rounded = round(input[index]);
    rounded = clamp(rounded, -128.0f, 127.0f);
    out[index] = char(rounded);
}
