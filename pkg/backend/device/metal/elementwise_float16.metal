#include <metal_stdlib>

using namespace metal;

template <typename BinaryOp>
static inline void binary_float16(
    device const half4* leftVector,
    device const half4* rightVector,
    device half4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    BinaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = op(leftVector[index], rightVector[index]);
        return;
    }

    device const half* left = reinterpret_cast<device const half*>(leftVector);
    device const half* right = reinterpret_cast<device const half*>(rightVector);
    device half* out = reinterpret_cast<device half*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = op(left[scalarIndex], right[scalarIndex]);
        }
    }
}

template <typename UnaryOp>
static inline void unary_float16(
    device const half4* inputVector,
    device half4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = op(inputVector[index]);
        return;
    }

    device const half* input = reinterpret_cast<device const half*>(inputVector);
    device half* out = reinterpret_cast<device half*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = op(input[scalarIndex]);
        }
    }
}

struct AddFloat16 {
    half4 operator()(half4 left, half4 right) const { return left + right; }
    half operator()(half left, half right) const { return left + right; }
};

struct SubFloat16 {
    half4 operator()(half4 left, half4 right) const { return left - right; }
    half operator()(half left, half right) const { return left - right; }
};

struct MulFloat16 {
    half4 operator()(half4 left, half4 right) const { return left * right; }
    half operator()(half left, half right) const { return left * right; }
};

struct DivFloat16 {
    half4 operator()(half4 left, half4 right) const { return left / right; }
    half operator()(half left, half right) const { return left / right; }
};

struct MaxFloat16 {
    half4 operator()(half4 left, half4 right) const { return max(left, right); }
    half operator()(half left, half right) const { return max(left, right); }
};

struct MinFloat16 {
    half4 operator()(half4 left, half4 right) const { return min(left, right); }
    half operator()(half left, half right) const { return min(left, right); }
};

struct EqFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left == right);
    }

    half operator()(half left, half right) const { return left == right ? 1.0h : 0.0h; }
};

struct NeFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left != right);
    }

    half operator()(half left, half right) const { return left != right ? 1.0h : 0.0h; }
};

struct LtFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left < right);
    }

    half operator()(half left, half right) const { return left < right ? 1.0h : 0.0h; }
};

struct LeFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left <= right);
    }

    half operator()(half left, half right) const { return left <= right ? 1.0h : 0.0h; }
};

struct GtFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left > right);
    }

    half operator()(half left, half right) const { return left > right ? 1.0h : 0.0h; }
};

struct GeFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left >= right);
    }

    half operator()(half left, half right) const { return left >= right ? 1.0h : 0.0h; }
};

struct ReluFloat16 {
    half4 operator()(half4 value) const { return max(value, half4(0.0h)); }
    half operator()(half value) const { return max(value, 0.0h); }
};

struct AbsFloat16 {
    half4 operator()(half4 value) const { return fabs(value); }
    half operator()(half value) const { return fabs(value); }
};

struct NegFloat16 {
    half4 operator()(half4 value) const { return -value; }
    half operator()(half value) const { return -value; }
};

struct SquareFloat16 {
    half4 operator()(half4 value) const { return value * value; }
    half operator()(half value) const { return value * value; }
};

struct RecipFloat16 {
    half4 operator()(half4 value) const { return half4(1.0h) / value; }
    half operator()(half value) const { return 1.0h / value; }
};

struct SqrtFloat16 {
    half4 operator()(half4 value) const { return sqrt(value); }
    half operator()(half value) const { return sqrt(value); }
};

struct SignFloat16 {
    half4 operator()(half4 value) const {
        half4 positive = select(half4(0.0h), half4(1.0h), value > half4(0.0h));
        return select(positive, half4(-1.0h), value < half4(0.0h));
    }

    half operator()(half value) const {
        if (value > 0.0h) {
            return 1.0h;
        }

        if (value < 0.0h) {
            return -1.0h;
        }

        return 0.0h;
    }
};

#define BINARY_FLOAT16_KERNEL(name, op) \
kernel void name##_float16( \
    device const half4* leftVector [[buffer(0)]], \
    device const half4* rightVector [[buffer(1)]], \
    device half4* outVector [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    binary_float16(leftVector, rightVector, outVector, count, index, op{}); \
}

#define UNARY_FLOAT16_KERNEL(name, op) \
kernel void name##_float16( \
    device const half4* inputVector [[buffer(0)]], \
    device half4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    unary_float16(inputVector, outVector, count, index, op{}); \
}

BINARY_FLOAT16_KERNEL(add, AddFloat16)
BINARY_FLOAT16_KERNEL(sub, SubFloat16)
BINARY_FLOAT16_KERNEL(mul, MulFloat16)
BINARY_FLOAT16_KERNEL(div, DivFloat16)
BINARY_FLOAT16_KERNEL(max, MaxFloat16)
BINARY_FLOAT16_KERNEL(min, MinFloat16)
BINARY_FLOAT16_KERNEL(eq, EqFloat16)
BINARY_FLOAT16_KERNEL(ne, NeFloat16)
BINARY_FLOAT16_KERNEL(lt, LtFloat16)
BINARY_FLOAT16_KERNEL(le, LeFloat16)
BINARY_FLOAT16_KERNEL(gt, GtFloat16)
BINARY_FLOAT16_KERNEL(ge, GeFloat16)

UNARY_FLOAT16_KERNEL(relu, ReluFloat16)
UNARY_FLOAT16_KERNEL(abs, AbsFloat16)
UNARY_FLOAT16_KERNEL(neg, NegFloat16)
UNARY_FLOAT16_KERNEL(square, SquareFloat16)
UNARY_FLOAT16_KERNEL(recip, RecipFloat16)
UNARY_FLOAT16_KERNEL(sqrt, SqrtFloat16)
UNARY_FLOAT16_KERNEL(sign, SignFloat16)
