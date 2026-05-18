#include <metal_stdlib>

using namespace metal;

template <typename BinaryOp>
static inline void binary_float32(
    device const float4* leftVector,
    device const float4* rightVector,
    device float4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    BinaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = op(leftVector[index], rightVector[index]);
        return;
    }

    device const float* left = reinterpret_cast<device const float*>(leftVector);
    device const float* right = reinterpret_cast<device const float*>(rightVector);
    device float* out = reinterpret_cast<device float*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = op(left[scalarIndex], right[scalarIndex]);
        }
    }
}

struct AddFloat32 {
    float4 operator()(float4 left, float4 right) const { return left + right; }
    float operator()(float left, float right) const { return left + right; }
};

struct SubFloat32 {
    float4 operator()(float4 left, float4 right) const { return left - right; }
    float operator()(float left, float right) const { return left - right; }
};

struct MulFloat32 {
    float4 operator()(float4 left, float4 right) const { return left * right; }
    float operator()(float left, float right) const { return left * right; }
};

struct DivFloat32 {
    float4 operator()(float4 left, float4 right) const { return left / right; }
    float operator()(float left, float right) const { return left / right; }
};

kernel void add_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, AddFloat32{});
}

kernel void sub_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, SubFloat32{});
}

kernel void mul_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, MulFloat32{});
}

kernel void div_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, DivFloat32{});
}
