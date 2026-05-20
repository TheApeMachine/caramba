package checkpoint

/*
EncodeFloat32DataNative writes src float32 payload into dst bytes using the best CPU kernel.
*/
func EncodeFloat32DataNative(dst []byte, src []float32) {
	if len(src) == 0 {
		return
	}

	encodeFloat32DataKernel(dst, src)
}

/*
DecodeFloat32DataNative reads float32 payload from src bytes into dst using the best CPU kernel.
*/
func DecodeFloat32DataNative(dst []float32, src []byte) {
	if len(dst) == 0 {
		return
	}

	decodeFloat32DataKernel(dst, src)
}
