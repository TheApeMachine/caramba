//go:build amd64

package checkpoint

//go:noescape
func CheckpointEncodeFloat32DataAVX512Asm(dst *byte, src *float32, count int)

//go:noescape
func CheckpointDecodeFloat32DataAVX512Asm(dst *float32, src *byte, count int)

/*
CheckpointEncodeFloat32DataAVX512 writes count float32 lanes from src into dst as little-endian bytes.
*/
func CheckpointEncodeFloat32DataAVX512(dst *byte, src *float32, count int) {
	if count == 0 {
		return
	}

	CheckpointEncodeFloat32DataAVX512Asm(dst, src, count)
}

/*
CheckpointDecodeFloat32DataAVX512 reads count float32 lanes from little-endian bytes in src into dst.
*/
func CheckpointDecodeFloat32DataAVX512(dst *float32, src *byte, count int) {
	if count == 0 {
		return
	}

	CheckpointDecodeFloat32DataAVX512Asm(dst, src, count)
}
