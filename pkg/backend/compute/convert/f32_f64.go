package convert

/*
Float32ToFloat64 widens float32 to float64. Dispatches to the
per-architecture native path (arm64 NEON FCVTL); falls back to the
scalar reference on other platforms.
*/
func Float32ToFloat64(dst []float64, src []float32) error {
	return float32ToFloat64Native(dst, src)
}

/*
Float64ToFloat32 narrows float64 to float32. Round-to-nearest-even.
*/
func Float64ToFloat32(dst []float32, src []float64) error {
	return float64ToFloat32Native(dst, src)
}
