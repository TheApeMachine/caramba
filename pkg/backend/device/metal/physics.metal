#include <metal_stdlib>

using namespace metal;

constant float physicsPi = 3.14159265358979323846f;

static inline float physics_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort physics_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32PhysicsStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16PhysicsStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16PhysicsStorage {
    static float load(device const ushort* values, uint index) {
        return physics_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = physics_float_to_bf16(value);
    }
};

template <typename Storage, typename Scalar>
static inline float physics_spacing(device const Scalar* spacing) {
    float dx = Storage::load(spacing, 0);
    return dx > 0.0f ? dx : 1.0f;
}

template <typename Storage, typename Scalar>
static inline void laplacian_kernel(
    device const Scalar* input,
    device const Scalar* spacing,
    device Scalar* out,
    constant uint& count,
    constant uint& rank,
    constant uint& dim0,
    constant uint& dim1,
    constant uint& dim2,
    uint index
) {
    if (index >= count) {
        return;
    }

    float dx = physics_spacing<Storage, Scalar>(spacing);
    float dxSquared = dx * dx;
    float center = Storage::load(input, index);

    if (rank == 1) {
        uint left = (index + dim0 - 1u) % dim0;
        uint right = (index + 1u) % dim0;
        float value = Storage::load(input, left) - 2.0f * center + Storage::load(input, right);
        Storage::store(out, index, value / dxSquared);
        return;
    }

    if (rank == 2) {
        uint row = index / dim1;
        uint col = index - row * dim1;
        uint up = ((row + dim0 - 1u) % dim0) * dim1 + col;
        uint down = ((row + 1u) % dim0) * dim1 + col;
        uint left = row * dim1 + ((col + dim1 - 1u) % dim1);
        uint right = row * dim1 + ((col + 1u) % dim1);
        float value = Storage::load(input, up) + Storage::load(input, down) +
            Storage::load(input, left) + Storage::load(input, right) - 4.0f * center;
        Storage::store(out, index, value / dxSquared);
        return;
    }

    uint plane = dim1 * dim2;
    uint depth = index / plane;
    uint rem = index - depth * plane;
    uint row = rem / dim2;
    uint col = rem - row * dim2;
    uint dm = ((depth + dim0 - 1u) % dim0) * plane + row * dim2 + col;
    uint dp = ((depth + 1u) % dim0) * plane + row * dim2 + col;
    uint rm = depth * plane + ((row + dim1 - 1u) % dim1) * dim2 + col;
    uint rp = depth * plane + ((row + 1u) % dim1) * dim2 + col;
    uint cm = depth * plane + row * dim2 + ((col + dim2 - 1u) % dim2);
    uint cp = depth * plane + row * dim2 + ((col + 1u) % dim2);
    float value = Storage::load(input, dm) + Storage::load(input, dp) +
        Storage::load(input, rm) + Storage::load(input, rp) +
        Storage::load(input, cm) + Storage::load(input, cp) - 6.0f * center;
    Storage::store(out, index, value / dxSquared);
}

template <typename Storage, typename Scalar>
static inline void laplacian4_kernel(
    device const Scalar* input,
    device const Scalar* spacing,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float dx = physics_spacing<Storage, Scalar>(spacing);
    float denominator = 12.0f * dx * dx;
    uint im2 = (index + count - 2u) % count;
    uint im1 = (index + count - 1u) % count;
    uint ip1 = (index + 1u) % count;
    uint ip2 = (index + 2u) % count;
    float value = -Storage::load(input, im2) + 16.0f * Storage::load(input, im1) -
        30.0f * Storage::load(input, index) + 16.0f * Storage::load(input, ip1) -
        Storage::load(input, ip2);
    Storage::store(out, index, value / denominator);
}

template <typename Storage, typename Scalar>
static inline void grad1d_kernel(
    device const Scalar* input,
    device const Scalar* spacing,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float dx = physics_spacing<Storage, Scalar>(spacing);
    uint left = (index + count - 1u) % count;
    uint right = (index + 1u) % count;
    Storage::store(out, index, (Storage::load(input, right) - Storage::load(input, left)) / (2.0f * dx));
}

template <typename Storage, typename Scalar>
static inline void quantum_potential_kernel(
    device const Scalar* density,
    device const Scalar* spacing,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    if (index == 0 || index + 1u == count) {
        Storage::store(out, index, 0.0f);
        return;
    }

    float rho = Storage::load(density, index);
    if (rho <= 1.0e-12f) {
        Storage::store(out, index, 0.0f);
        return;
    }

    float dx = physics_spacing<Storage, Scalar>(spacing);
    float sqrtRho = sqrt(rho);
    float sqrtLeft = sqrt(max(1.0e-12f, Storage::load(density, index - 1u)));
    float sqrtRight = sqrt(max(1.0e-12f, Storage::load(density, index + 1u)));
    float laplacian = (sqrtRight - 2.0f * sqrtRho + sqrtLeft) / (dx * dx);
    Storage::store(out, index, -0.5f * laplacian / sqrtRho);
}

template <typename Storage, typename Scalar>
static inline void bohmian_velocity_kernel(
    device const Scalar* phase,
    device const Scalar* spacing,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    if (index == 0 || index + 1u == count) {
        Storage::store(out, index, 0.0f);
        return;
    }

    float dx = physics_spacing<Storage, Scalar>(spacing);
    Storage::store(out, index, (Storage::load(phase, index + 1u) - Storage::load(phase, index - 1u)) / (2.0f * dx));
}

template <typename Storage, typename Scalar>
static inline void madelung_continuity_kernel(
    device const Scalar* density,
    device const Scalar* velocity,
    device const Scalar* spacing,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    if (index == 0 || index + 1u == count) {
        Storage::store(out, index, 0.0f);
        return;
    }

    float dx = physics_spacing<Storage, Scalar>(spacing);
    float fluxRight = Storage::load(density, index + 1u) * Storage::load(velocity, index + 1u);
    float fluxLeft = Storage::load(density, index - 1u) * Storage::load(velocity, index - 1u);
    Storage::store(out, index, (fluxRight - fluxLeft) / (2.0f * dx));
}

template <typename Storage, typename Scalar>
static inline void fft_bit_reverse_kernel(
    device const Scalar* realIn,
    device const Scalar* imagIn,
    device Scalar* realOut,
    device Scalar* imagOut,
    constant uint& count,
    constant uint& bits,
    uint index
) {
    if (index >= count) {
        return;
    }

    uint reversed = reverse_bits(index) >> (32u - bits);
    Storage::store(realOut, reversed, Storage::load(realIn, index));
    Storage::store(imagOut, reversed, Storage::load(imagIn, index));
}

template <typename Storage, typename Scalar>
static inline void fft_stage_kernel(
    device Scalar* realValues,
    device Scalar* imagValues,
    constant uint& length,
    constant uint& inverseValue,
    uint butterfly
) {
    uint halfLength = length >> 1u;
    uint block = butterfly / halfLength;
    uint offset = butterfly - block * halfLength;
    uint upper = block * length + offset;
    uint lower = upper + halfLength;
    float sign = inverseValue != 0u ? 1.0f : -1.0f;
    float angle = sign * 2.0f * physicsPi / float(length);
    float stepReal = cos(angle);
    float stepImag = sin(angle);
    float twiddleReal = 1.0f;
    float twiddleImag = 0.0f;

    for (uint step = 0; step < offset; step++) {
        float nextReal = twiddleReal * stepReal - twiddleImag * stepImag;
        float nextImag = twiddleReal * stepImag + twiddleImag * stepReal;
        twiddleReal = nextReal;
        twiddleImag = nextImag;
    }

    float lowerReal = Storage::load(realValues, lower);
    float lowerImag = Storage::load(imagValues, lower);
    float tempReal = twiddleReal * lowerReal - twiddleImag * lowerImag;
    float tempImag = twiddleReal * lowerImag + twiddleImag * lowerReal;
    float upperReal = Storage::load(realValues, upper);
    float upperImag = Storage::load(imagValues, upper);

    Storage::store(realValues, lower, upperReal - tempReal);
    Storage::store(imagValues, lower, upperImag - tempImag);
    Storage::store(realValues, upper, upperReal + tempReal);
    Storage::store(imagValues, upper, upperImag + tempImag);
}

template <typename Storage, typename Scalar>
static inline void fft_scale_kernel(
    device Scalar* realValues,
    device Scalar* imagValues,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float scale = 1.0f / float(count);
    Storage::store(realValues, index, Storage::load(realValues, index) * scale);
    Storage::store(imagValues, index, Storage::load(imagValues, index) * scale);
}

template <typename Storage, typename Scalar>
static inline void dft_naive_kernel(
    device const Scalar* realIn,
    device const Scalar* imagIn,
    device Scalar* realOut,
    device Scalar* imagOut,
    constant uint& count,
    constant uint& inverseValue,
    uint index
) {
    if (index >= count) {
        return;
    }

    float sign = inverseValue != 0u ? 1.0f : -1.0f;
    float sumReal = 0.0f;
    float sumImag = 0.0f;

    for (uint source = 0; source < count; source++) {
        float angle = sign * 2.0f * physicsPi * float(index) * float(source) / float(count);
        float c = cos(angle);
        float s = sin(angle);
        float realValue = Storage::load(realIn, source);
        float imagValue = Storage::load(imagIn, source);
        sumReal += realValue * c - imagValue * s;
        sumImag += realValue * s + imagValue * c;
    }

    if (inverseValue != 0u) {
        float scale = 1.0f / float(count);
        sumReal *= scale;
        sumImag *= scale;
    }

    Storage::store(realOut, index, sumReal);
    Storage::store(imagOut, index, sumImag);
}

#define PHYSICS_LAPLACIAN_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* spacing [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    constant uint& rank [[buffer(4)]], \
    constant uint& dim0 [[buffer(5)]], \
    constant uint& dim1 [[buffer(6)]], \
    constant uint& dim2 [[buffer(7)]], \
    uint index [[thread_position_in_grid]] \
) { \
    laplacian_kernel<storage, scalar>(input, spacing, out, count, rank, dim0, dim1, dim2, index); \
}

#define PHYSICS_VECTOR_KERNEL(name, helper, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* spacing [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    helper<storage, scalar>(input, spacing, out, count, index); \
}

#define PHYSICS_MADELUNG_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* density [[buffer(0)]], \
    device const scalar* velocity [[buffer(1)]], \
    device const scalar* spacing [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    madelung_continuity_kernel<storage, scalar>(density, velocity, spacing, out, count, index); \
}

#define PHYSICS_FFT_KERNELS(prefix, storage, scalar) \
kernel void prefix##_fft_bit_reverse( \
    device const scalar* realIn [[buffer(0)]], \
    device const scalar* imagIn [[buffer(1)]], \
    device scalar* realOut [[buffer(2)]], \
    device scalar* imagOut [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    constant uint& bits [[buffer(5)]], \
    uint index [[thread_position_in_grid]] \
) { \
    fft_bit_reverse_kernel<storage, scalar>(realIn, imagIn, realOut, imagOut, count, bits, index); \
} \
kernel void prefix##_fft_stage( \
    device scalar* realValues [[buffer(0)]], \
    device scalar* imagValues [[buffer(1)]], \
    constant uint& length [[buffer(2)]], \
    constant uint& inverseValue [[buffer(3)]], \
    uint butterfly [[thread_position_in_grid]] \
) { \
    fft_stage_kernel<storage, scalar>(realValues, imagValues, length, inverseValue, butterfly); \
} \
kernel void prefix##_fft_scale( \
    device scalar* realValues [[buffer(0)]], \
    device scalar* imagValues [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    fft_scale_kernel<storage, scalar>(realValues, imagValues, count, index); \
} \
kernel void prefix##_dft_naive( \
    device const scalar* realIn [[buffer(0)]], \
    device const scalar* imagIn [[buffer(1)]], \
    device scalar* realOut [[buffer(2)]], \
    device scalar* imagOut [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    constant uint& inverseValue [[buffer(5)]], \
    uint index [[thread_position_in_grid]] \
) { \
    dft_naive_kernel<storage, scalar>(realIn, imagIn, realOut, imagOut, count, inverseValue, index); \
}

PHYSICS_LAPLACIAN_KERNEL(laplacian_float32, Float32PhysicsStorage, float)
PHYSICS_VECTOR_KERNEL(laplacian4_float32, laplacian4_kernel, Float32PhysicsStorage, float)
PHYSICS_VECTOR_KERNEL(grad1d_float32, grad1d_kernel, Float32PhysicsStorage, float)
PHYSICS_VECTOR_KERNEL(divergence1d_float32, grad1d_kernel, Float32PhysicsStorage, float)
PHYSICS_VECTOR_KERNEL(quantum_potential_float32, quantum_potential_kernel, Float32PhysicsStorage, float)
PHYSICS_VECTOR_KERNEL(bohmian_velocity_float32, bohmian_velocity_kernel, Float32PhysicsStorage, float)
PHYSICS_MADELUNG_KERNEL(madelung_continuity_float32, Float32PhysicsStorage, float)
PHYSICS_FFT_KERNELS(float32, Float32PhysicsStorage, float)

PHYSICS_LAPLACIAN_KERNEL(laplacian_float16, Float16PhysicsStorage, half)
PHYSICS_VECTOR_KERNEL(laplacian4_float16, laplacian4_kernel, Float16PhysicsStorage, half)
PHYSICS_VECTOR_KERNEL(grad1d_float16, grad1d_kernel, Float16PhysicsStorage, half)
PHYSICS_VECTOR_KERNEL(divergence1d_float16, grad1d_kernel, Float16PhysicsStorage, half)
PHYSICS_VECTOR_KERNEL(quantum_potential_float16, quantum_potential_kernel, Float16PhysicsStorage, half)
PHYSICS_VECTOR_KERNEL(bohmian_velocity_float16, bohmian_velocity_kernel, Float16PhysicsStorage, half)
PHYSICS_MADELUNG_KERNEL(madelung_continuity_float16, Float16PhysicsStorage, half)
PHYSICS_FFT_KERNELS(float16, Float16PhysicsStorage, half)

PHYSICS_LAPLACIAN_KERNEL(laplacian_bfloat16, BFloat16PhysicsStorage, ushort)
PHYSICS_VECTOR_KERNEL(laplacian4_bfloat16, laplacian4_kernel, BFloat16PhysicsStorage, ushort)
PHYSICS_VECTOR_KERNEL(grad1d_bfloat16, grad1d_kernel, BFloat16PhysicsStorage, ushort)
PHYSICS_VECTOR_KERNEL(divergence1d_bfloat16, grad1d_kernel, BFloat16PhysicsStorage, ushort)
PHYSICS_VECTOR_KERNEL(quantum_potential_bfloat16, quantum_potential_kernel, BFloat16PhysicsStorage, ushort)
PHYSICS_VECTOR_KERNEL(bohmian_velocity_bfloat16, bohmian_velocity_kernel, BFloat16PhysicsStorage, ushort)
PHYSICS_MADELUNG_KERNEL(madelung_continuity_bfloat16, BFloat16PhysicsStorage, ushort)
PHYSICS_FFT_KERNELS(bfloat16, BFloat16PhysicsStorage, ushort)
