package checkpoint

import (
	"encoding/binary"
	"math"
)

/*
encodeFloat32DataScalar writes src float32 values into dst as little-endian uint32 bytes.
*/
func encodeFloat32DataScalar(dst []byte, src []float32) {
	for index, value := range src {
		binary.LittleEndian.PutUint32(dst[index*4:], math.Float32bits(value))
	}
}

/*
decodeFloat32DataScalar reads little-endian uint32 bytes from src into dst float32 values.
*/
func decodeFloat32DataScalar(dst []float32, src []byte) {
	for index := range dst {
		dst[index] = math.Float32frombits(binary.LittleEndian.Uint32(src[index*4:]))
	}
}
