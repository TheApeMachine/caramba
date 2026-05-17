package dtype

import (
	"encoding/binary"
	"math"
)

/*
BF16 is a 16-bit floating-point number.
*/
type BF16 uint16

/*
NewFromFloat32 creates a BF16 from a float32.
*/
func NewBfloat16FromFloat32(value float32) BF16 {
	return BF16(math.Float32bits(value) >> 16)
}

/*
NewBfloat16FromBytes creates a BF16 from a byte slice.
*/
func NewBfloat16FromBytes(buf []byte) BF16 {
	return BF16(binary.BigEndian.Uint16(buf[:2]))
}

/*
Bytes returns the bytes of the BF16.
*/
func (bf16 *BF16) Bytes() []byte {
	var buf [2]byte
	binary.BigEndian.PutUint16(buf[:], uint16(*bf16))

	return buf[:]
}

/*
Decode decodes a byte slice into a slice of BF16.
*/
func (bf16 *BF16) Decode(buf []byte) []BF16 {
	var out []BF16

	for index := 0; index < len(buf); index += 2 {
		out = append(out, NewBfloat16FromBytes(buf[index:]))
	}

	return out
}

/*
Encode encodes a slice of BF16 into a byte slice.
*/
func (bf16 *BF16) Encode(values []BF16) []byte {
	var out []byte

	for _, value := range values {
		out = append(out, value.Bytes()...)
	}

	return out
}

/*
DecodeFloat32 decodes a byte slice into a slice of float32.
*/
func (bf16 *BF16) DecodeFloat32(buf []byte) []float32 {
	var out []float32

	for index := 0; index < len(buf); index += 2 {
		value := NewBfloat16FromBytes(buf[index:])
		out = append(out, (&value).Float32())
	}

	return out
}

/*
EncodeFloat32 encodes a slice of float32 into a byte slice.
*/
func (bf16 *BF16) EncodeFloat32(values []float32) []byte {
	var out []byte

	for _, value := range values {
		bfloat16 := NewBfloat16FromFloat32(value)
		out = append(out, (&bfloat16).Bytes()...)
	}

	return out
}

/*
Float32 returns the float32 value of the BF16.
*/
func (bf16 *BF16) Float32() float32 {
	return math.Float32frombits(uint32(*bf16) << 16)
}
