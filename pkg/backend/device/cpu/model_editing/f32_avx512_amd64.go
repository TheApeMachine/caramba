//go:build amd64

package model_editing

//go:noescape
func WeightGraftAddFloat32AVX512Asm(weights, injection *float32, count int)

func WeightGraftAddFloat32AVX512(weights, injection *float32, count int) {
	if count == 0 {
		return
	}

	WeightGraftAddFloat32AVX512Asm(weights, injection, count)
}
