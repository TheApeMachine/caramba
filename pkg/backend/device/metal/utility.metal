#include <metal_stdlib>

using namespace metal;

static inline void write_u64_le(device uchar* out, uint offset, ulong value) {
    for (uint byteIndex = 0; byteIndex < 8; byteIndex++) {
        out[offset + byteIndex] = uchar((value >> (byteIndex * 8)) & 0xfful);
    }
}

static inline void write_u32_le(device uchar* out, uint offset, uint value) {
    for (uint byteIndex = 0; byteIndex < 4; byteIndex++) {
        out[offset + byteIndex] = uchar((value >> (byteIndex * 8)) & 0xffu);
    }
}

static inline bool mask_bit(device const uchar* mask, uint index) {
    return ((mask[index >> 3u] >> (index & 7u)) & 1u) != 0u;
}

kernel void checkpoint_encode_float32(
    device const uint* inputBits [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant uint& rank [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant ulong* dims [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    uint headerBytes = 16u + rank * 8u;

    if (index == 0u) {
        write_u64_le(out, 0u, ulong(rank));
        write_u64_le(out, 8u, ulong(count) * 4ul);

        for (uint dimIndex = 0u; dimIndex < rank; dimIndex++) {
            write_u64_le(out, 16u + dimIndex * 8u, dims[dimIndex]);
        }
    }

    if (index >= count) {
        return;
    }

    write_u32_le(out, headerBytes + index * 4u, inputBits[index]);
}

kernel void checkpoint_decode_float32(
    device const uchar* input [[buffer(0)]],
    device uint* outBits [[buffer(1)]],
    constant uint& headerBytes [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    device const uint* inputBits = reinterpret_cast<device const uint*>(input + headerBytes);
    outBits[index] = inputBits[index];
}

kernel void tokenizer_pack_int32(
    device const int4* input [[buffer(0)]],
    device int4* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4u;

    if (base + 3u < count) {
        out[index] = input[index];
        return;
    }

    device const int* inputScalar = reinterpret_cast<device const int*>(input);
    device int* outScalar = reinterpret_cast<device int*>(out);

    for (uint offset = 0u; offset < 4u; offset++) {
        uint elementIndex = base + offset;

        if (elementIndex < count) {
            outScalar[elementIndex] = inputScalar[elementIndex];
        }
    }
}

kernel void weight_freeze_mask_float32(
    device const uchar* mask [[buffer(0)]],
    device const float4* gradients [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4u;
    float4 values = gradients[index];
    float4 result = float4(0.0f);

    if (base < count && mask_bit(mask, base)) {
        result.x = values.x;
    }

    if (base + 1u < count && mask_bit(mask, base + 1u)) {
        result.y = values.y;
    }

    if (base + 2u < count && mask_bit(mask, base + 2u)) {
        result.z = values.z;
    }

    if (base + 3u < count && mask_bit(mask, base + 3u)) {
        result.w = values.w;
    }

    out[index] = result;
}

kernel void weight_freeze_mask_float16(
    device const uchar* mask [[buffer(0)]],
    device const half4* gradients [[buffer(1)]],
    device half4* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4u;
    half4 values = gradients[index];
    half4 result = half4(0.0h);

    if (base < count && mask_bit(mask, base)) {
        result.x = values.x;
    }

    if (base + 1u < count && mask_bit(mask, base + 1u)) {
        result.y = values.y;
    }

    if (base + 2u < count && mask_bit(mask, base + 2u)) {
        result.z = values.z;
    }

    if (base + 3u < count && mask_bit(mask, base + 3u)) {
        result.w = values.w;
    }

    out[index] = result;
}

kernel void weight_freeze_mask_bfloat16(
    device const uchar* mask [[buffer(0)]],
    device const ushort4* gradients [[buffer(1)]],
    device ushort4* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4u;
    ushort4 values = gradients[index];
    ushort4 result = ushort4(0u);

    if (base < count && mask_bit(mask, base)) {
        result.x = values.x;
    }

    if (base + 1u < count && mask_bit(mask, base + 1u)) {
        result.y = values.y;
    }

    if (base + 2u < count && mask_bit(mask, base + 2u)) {
        result.z = values.z;
    }

    if (base + 3u < count && mask_bit(mask, base + 3u)) {
        result.w = values.w;
    }

    out[index] = result;
}
