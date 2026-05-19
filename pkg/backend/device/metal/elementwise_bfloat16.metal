#include <metal_stdlib>

using namespace metal;

static inline float bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline float4 bf16_to_float4(ushort4 value) {
    return float4(
        bf16_to_float(value.x),
        bf16_to_float(value.y),
        bf16_to_float(value.z),
        bf16_to_float(value.w)
    );
}

static inline ushort4 float4_to_bf16(float4 value) {
    return ushort4(
        float_to_bf16(value.x),
        float_to_bf16(value.y),
        float_to_bf16(value.z),
        float_to_bf16(value.w)
    );
}

template <typename BinaryOp>
static inline void binary_bfloat16(
    device const ushort4* leftVector,
    device const ushort4* rightVector,
    device ushort4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    BinaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = float4_to_bf16(op(
            bf16_to_float4(leftVector[index]),
            bf16_to_float4(rightVector[index])
        ));
        return;
    }

    device const ushort* left = reinterpret_cast<device const ushort*>(leftVector);
    device const ushort* right = reinterpret_cast<device const ushort*>(rightVector);
    device ushort* out = reinterpret_cast<device ushort*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = float_to_bf16(op(
                bf16_to_float(left[scalarIndex]),
                bf16_to_float(right[scalarIndex])
            ));
        }
    }
}

template <typename UnaryOp>
static inline void unary_bfloat16(
    device const ushort4* inputVector,
    device ushort4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = float4_to_bf16(op(bf16_to_float4(inputVector[index])));
        return;
    }

    device const ushort* input = reinterpret_cast<device const ushort*>(inputVector);
    device ushort* out = reinterpret_cast<device ushort*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = float_to_bf16(op(bf16_to_float(input[scalarIndex])));
        }
    }
}

struct AddBFloat16 {
    float4 operator()(float4 left, float4 right) const { return left + right; }
    float operator()(float left, float right) const { return left + right; }
};

struct SubBFloat16 {
    float4 operator()(float4 left, float4 right) const { return left - right; }
    float operator()(float left, float right) const { return left - right; }
};

struct MulBFloat16 {
    float4 operator()(float4 left, float4 right) const { return left * right; }
    float operator()(float left, float right) const { return left * right; }
};

struct DivBFloat16 {
    float4 operator()(float4 left, float4 right) const { return left / right; }
    float operator()(float left, float right) const { return left / right; }
};

struct MaxBFloat16 {
    float4 operator()(float4 left, float4 right) const { return max(left, right); }
    float operator()(float left, float right) const { return max(left, right); }
};

struct MinBFloat16 {
    float4 operator()(float4 left, float4 right) const { return min(left, right); }
    float operator()(float left, float right) const { return min(left, right); }
};

struct EqBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left == right);
    }

    float operator()(float left, float right) const { return left == right ? 1.0 : 0.0; }
};

struct NeBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left != right);
    }

    float operator()(float left, float right) const { return left != right ? 1.0 : 0.0; }
};

struct LtBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left < right);
    }

    float operator()(float left, float right) const { return left < right ? 1.0 : 0.0; }
};

struct LeBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left <= right);
    }

    float operator()(float left, float right) const { return left <= right ? 1.0 : 0.0; }
};

struct GtBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left > right);
    }

    float operator()(float left, float right) const { return left > right ? 1.0 : 0.0; }
};

struct GeBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left >= right);
    }

    float operator()(float left, float right) const { return left >= right ? 1.0 : 0.0; }
};

struct PowBFloat16 {
    float4 operator()(float4 left, float4 right) const { return pow(left, right); }
    float operator()(float left, float right) const { return pow(left, right); }
};

struct Atan2BFloat16 {
    float4 operator()(float4 left, float4 right) const { return atan2(left, right); }
    float operator()(float left, float right) const { return atan2(left, right); }
};

struct ModBFloat16 {
    float4 operator()(float4 left, float4 right) const { return fmod(left, right); }
    float operator()(float left, float right) const { return fmod(left, right); }
};

struct ReluBFloat16 {
    float4 operator()(float4 value) const { return max(value, float4(0.0)); }
    float operator()(float value) const { return max(value, 0.0); }
};

struct AbsBFloat16 {
    float4 operator()(float4 value) const { return fabs(value); }
    float operator()(float value) const { return fabs(value); }
};

struct NegBFloat16 {
    float4 operator()(float4 value) const { return -value; }
    float operator()(float value) const { return -value; }
};

struct SquareBFloat16 {
    float4 operator()(float4 value) const { return value * value; }
    float operator()(float value) const { return value * value; }
};

struct RecipBFloat16 {
    float4 operator()(float4 value) const { return float4(1.0) / value; }
    float operator()(float value) const { return 1.0 / value; }
};

struct SqrtBFloat16 {
    float4 operator()(float4 value) const { return sqrt(value); }
    float operator()(float value) const { return sqrt(value); }
};

struct SignBFloat16 {
    float4 operator()(float4 value) const {
        float4 positive = select(float4(0.0), float4(1.0), value > float4(0.0));
        return select(positive, float4(-1.0), value < float4(0.0));
    }

    float operator()(float value) const {
        if (value > 0.0) {
            return 1.0;
        }

        if (value < 0.0) {
            return -1.0;
        }

        return 0.0;
    }
};

#define BINARY_BFLOAT16_KERNEL(name, op) \
kernel void name##_bfloat16( \
    device const ushort4* leftVector [[buffer(0)]], \
    device const ushort4* rightVector [[buffer(1)]], \
    device ushort4* outVector [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    binary_bfloat16(leftVector, rightVector, outVector, count, index, op{}); \
}

#define UNARY_BFLOAT16_KERNEL(name, op) \
kernel void name##_bfloat16( \
    device const ushort4* inputVector [[buffer(0)]], \
    device ushort4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    unary_bfloat16(inputVector, outVector, count, index, op{}); \
}

BINARY_BFLOAT16_KERNEL(add, AddBFloat16)
BINARY_BFLOAT16_KERNEL(sub, SubBFloat16)
BINARY_BFLOAT16_KERNEL(mul, MulBFloat16)
BINARY_BFLOAT16_KERNEL(div, DivBFloat16)
BINARY_BFLOAT16_KERNEL(max, MaxBFloat16)
BINARY_BFLOAT16_KERNEL(min, MinBFloat16)
BINARY_BFLOAT16_KERNEL(eq, EqBFloat16)
BINARY_BFLOAT16_KERNEL(ne, NeBFloat16)
BINARY_BFLOAT16_KERNEL(lt, LtBFloat16)
BINARY_BFLOAT16_KERNEL(le, LeBFloat16)
BINARY_BFLOAT16_KERNEL(gt, GtBFloat16)
BINARY_BFLOAT16_KERNEL(ge, GeBFloat16)
BINARY_BFLOAT16_KERNEL(pow, PowBFloat16)
BINARY_BFLOAT16_KERNEL(atan2, Atan2BFloat16)
BINARY_BFLOAT16_KERNEL(mod, ModBFloat16)

UNARY_BFLOAT16_KERNEL(relu, ReluBFloat16)
UNARY_BFLOAT16_KERNEL(abs, AbsBFloat16)
UNARY_BFLOAT16_KERNEL(neg, NegBFloat16)
UNARY_BFLOAT16_KERNEL(square, SquareBFloat16)
UNARY_BFLOAT16_KERNEL(recip, RecipBFloat16)
UNARY_BFLOAT16_KERNEL(sqrt, SqrtBFloat16)
UNARY_BFLOAT16_KERNEL(sign, SignBFloat16)
