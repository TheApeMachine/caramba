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

template <typename UnaryOp>
static inline void unary_float32(
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

struct MaxFloat32 {
    float4 operator()(float4 left, float4 right) const { return max(left, right); }
    float operator()(float left, float right) const { return max(left, right); }
};

struct MinFloat32 {
    float4 operator()(float4 left, float4 right) const { return min(left, right); }
    float operator()(float left, float right) const { return min(left, right); }
};

struct EqFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left == right);
    }

    float operator()(float left, float right) const { return left == right ? 1.0 : 0.0; }
};

struct NeFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left != right);
    }

    float operator()(float left, float right) const { return left != right ? 1.0 : 0.0; }
};

struct LtFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left < right);
    }

    float operator()(float left, float right) const { return left < right ? 1.0 : 0.0; }
};

struct LeFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left <= right);
    }

    float operator()(float left, float right) const { return left <= right ? 1.0 : 0.0; }
};

struct GtFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left > right);
    }

    float operator()(float left, float right) const { return left > right ? 1.0 : 0.0; }
};

struct GeFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left >= right);
    }

    float operator()(float left, float right) const { return left >= right ? 1.0 : 0.0; }
};

struct ReluFloat32 {
    float4 operator()(float4 value) const { return max(value, float4(0.0)); }
    float operator()(float value) const { return max(value, 0.0); }
};

struct AbsFloat32 {
    float4 operator()(float4 value) const { return fabs(value); }
    float operator()(float value) const { return fabs(value); }
};

struct NegFloat32 {
    float4 operator()(float4 value) const { return -value; }
    float operator()(float value) const { return -value; }
};

struct SquareFloat32 {
    float4 operator()(float4 value) const { return value * value; }
    float operator()(float value) const { return value * value; }
};

struct RecipFloat32 {
    float4 operator()(float4 value) const { return float4(1.0) / value; }
    float operator()(float value) const { return 1.0 / value; }
};

struct SqrtFloat32 {
    float4 operator()(float4 value) const { return sqrt(value); }
    float operator()(float value) const { return sqrt(value); }
};

struct SignFloat32 {
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

kernel void max_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, MaxFloat32{});
}

kernel void min_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, MinFloat32{});
}

kernel void eq_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, EqFloat32{});
}

kernel void ne_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, NeFloat32{});
}

kernel void lt_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, LtFloat32{});
}

kernel void le_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, LeFloat32{});
}

kernel void gt_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, GtFloat32{});
}

kernel void ge_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, GeFloat32{});
}

kernel void relu_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    unary_float32(inputVector, outVector, count, index, ReluFloat32{});
}

kernel void abs_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    unary_float32(inputVector, outVector, count, index, AbsFloat32{});
}

kernel void neg_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    unary_float32(inputVector, outVector, count, index, NegFloat32{});
}

kernel void square_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    unary_float32(inputVector, outVector, count, index, SquareFloat32{});
}

kernel void recip_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    unary_float32(inputVector, outVector, count, index, RecipFloat32{});
}

kernel void sqrt_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    unary_float32(inputVector, outVector, count, index, SqrtFloat32{});
}

kernel void sign_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    unary_float32(inputVector, outVector, count, index, SignFloat32{});
}
