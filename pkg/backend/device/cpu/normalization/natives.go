package normalization

import "github.com/theapemachine/caramba/pkg/backend/device/cpu/reduction"

func SumFloat32Native(values []float32) float32 {
	return reduction.SumFloat32Native(values)
}
