#include <metal_stdlib>

using namespace metal;

constant float metalSELUAlpha = 1.67326324235437728482f;
constant float metalSELUScale = 1.05070098735548049342f;
constant float metalLeakyReLUSlope = 0.01f;

static inline float extended_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort extended_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline float4 extended_bf16_to_float4(ushort4 value) {
    return float4(
        extended_bf16_to_float(value.x),
        extended_bf16_to_float(value.y),
        extended_bf16_to_float(value.z),
        extended_bf16_to_float(value.w)
    );
}

static inline ushort4 extended_float4_to_bf16(float4 value) {
    return ushort4(
        extended_float_to_bf16(value.x),
        extended_float_to_bf16(value.y),
        extended_float_to_bf16(value.z),
        extended_float_to_bf16(value.w)
    );
}

template <typename UnaryOp>
static inline void extended_unary_float32(
    device const float4* inputVector,
    device float4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = op(inputVector[index]);
        return;
    }

    device const float* input = reinterpret_cast<device const float*>(inputVector);
    device float* out = reinterpret_cast<device float*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = op(input[scalarIndex]);
        }
    }
}

template <typename UnaryOp>
static inline void extended_unary_float16(
    device const half4* inputVector,
    device half4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = half4(op(float4(inputVector[index])));
        return;
    }

    device const half* input = reinterpret_cast<device const half*>(inputVector);
    device half* out = reinterpret_cast<device half*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = half(op(float(input[scalarIndex])));
        }
    }
}

template <typename UnaryOp>
static inline void extended_unary_bfloat16(
    device const ushort4* inputVector,
    device ushort4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = extended_float4_to_bf16(op(extended_bf16_to_float4(inputVector[index])));
        return;
    }

    device const ushort* input = reinterpret_cast<device const ushort*>(inputVector);
    device ushort* out = reinterpret_cast<device ushort*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = extended_float_to_bf16(op(extended_bf16_to_float(input[scalarIndex])));
        }
    }
}

struct RsqrtOp {
    float4 operator()(float4 value) const { return float4(1.0f) / sqrt(value); }
    float operator()(float value) const { return 1.0f / sqrt(value); }
};

struct ExpOp {
    float4 operator()(float4 value) const { return precise::exp(value); }
    float operator()(float value) const { return precise::exp(value); }
};

struct LogOp {
    float4 operator()(float4 value) const { return precise::log(value); }
    float operator()(float value) const { return precise::log(value); }
};

struct SinOp {
    float4 operator()(float4 value) const { return precise::sin(value); }
    float operator()(float value) const { return precise::sin(value); }
};

struct CosOp {
    float4 operator()(float4 value) const { return precise::cos(value); }
    float operator()(float value) const { return precise::cos(value); }
};

struct TanhOp {
    float4 operator()(float4 value) const { return precise::tanh(value); }
    float operator()(float value) const { return precise::tanh(value); }
};

struct SigmoidOp {
    float4 operator()(float4 value) const { return float4(1.0f) / (float4(1.0f) + precise::exp(-value)); }
    float operator()(float value) const { return 1.0f / (1.0f + precise::exp(-value)); }
};

struct SiluOp {
    float4 operator()(float4 value) const { return value / (float4(1.0f) + precise::exp(-value)); }
    float operator()(float value) const { return value / (1.0f + precise::exp(-value)); }
};

struct SoftsignOp {
    float4 operator()(float4 value) const { return value / (float4(1.0f) + fabs(value)); }
    float operator()(float value) const { return value / (1.0f + fabs(value)); }
};

struct ELUOp {
    float4 operator()(float4 value) const {
        return select(precise::exp(value) - float4(1.0f), value, value > float4(0.0f));
    }

    float operator()(float value) const {
        return value > 0.0f ? value : precise::exp(value) - 1.0f;
    }
};

struct SELUOp {
    float4 operator()(float4 value) const {
        float4 negative = metalSELUScale * metalSELUAlpha * (precise::exp(value) - float4(1.0f));
        return select(negative, metalSELUScale * value, value > float4(0.0f));
    }

    float operator()(float value) const {
        if (value > 0.0f) {
            return metalSELUScale * value;
        }

        return metalSELUScale * metalSELUAlpha * (precise::exp(value) - 1.0f);
    }
};

struct LeakyReLUOp {
    float4 operator()(float4 value) const {
        return select(metalLeakyReLUSlope * value, value, value > float4(0.0f));
    }

    float operator()(float value) const {
        return value > 0.0f ? value : metalLeakyReLUSlope * value;
    }
};

struct HardSigmoidOp {
    float4 operator()(float4 value) const {
        return clamp(value / float4(6.0f) + float4(0.5f), float4(0.0f), float4(1.0f));
    }

    float operator()(float value) const {
        return clamp(value / 6.0f + 0.5f, 0.0f, 1.0f);
    }
};

struct HardSwishOp {
    float4 operator()(float4 value) const {
        return value * clamp((value + float4(3.0f)) / float4(6.0f), float4(0.0f), float4(1.0f));
    }

    float operator()(float value) const {
        return value * clamp((value + 3.0f) / 6.0f, 0.0f, 1.0f);
    }
};

#define EXTENDED_UNARY_FLOAT32_KERNEL(name, op) \
kernel void name##_float32( \
    device const float4* inputVector [[buffer(0)]], \
    device float4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    extended_unary_float32(inputVector, outVector, count, index, op{}); \
}

#define EXTENDED_UNARY_FLOAT16_KERNEL(name, op) \
kernel void name##_float16( \
    device const half4* inputVector [[buffer(0)]], \
    device half4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    extended_unary_float16(inputVector, outVector, count, index, op{}); \
}

#define EXTENDED_UNARY_BFLOAT16_KERNEL(name, op) \
kernel void name##_bfloat16( \
    device const ushort4* inputVector [[buffer(0)]], \
    device ushort4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    extended_unary_bfloat16(inputVector, outVector, count, index, op{}); \
}

#define EXTENDED_UNARY_KERNELS(dtype_macro) \
dtype_macro(rsqrt, RsqrtOp) \
dtype_macro(exp, ExpOp) \
dtype_macro(log, LogOp) \
dtype_macro(sin, SinOp) \
dtype_macro(cos, CosOp) \
dtype_macro(tanh, TanhOp) \
dtype_macro(sigmoid, SigmoidOp) \
dtype_macro(silu, SiluOp) \
dtype_macro(swish, SiluOp) \
dtype_macro(softsign, SoftsignOp) \
dtype_macro(elu, ELUOp) \
dtype_macro(selu, SELUOp) \
dtype_macro(leaky_relu, LeakyReLUOp) \
dtype_macro(hardsigmoid, HardSigmoidOp) \
dtype_macro(hardswish, HardSwishOp)

EXTENDED_UNARY_KERNELS(EXTENDED_UNARY_FLOAT32_KERNEL)
EXTENDED_UNARY_KERNELS(EXTENDED_UNARY_FLOAT16_KERNEL)
EXTENDED_UNARY_KERNELS(EXTENDED_UNARY_BFLOAT16_KERNEL)
