package neon

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Mixed-precision losses. All 6 losses (mse, mae, huber, bce, kl, ce)
take bf16/fp16 inputs, accumulate in f32 per §5.5, and write the
scalar result narrowed back to the input dtype.
*/

func init() {
	for _, paramDType := range []dtype.DType{dtype.BFloat16, dtype.Float16} {
		paramDType := paramDType

		// MSE / MAE / Huber / BCE / KL: signature is (predictions, targets, output)
		// with all three at paramDType.
		for _, name := range []string{
			"mse_loss", "mae_loss", "huber_loss",
			"binary_cross_entropy", "kl_divergence",
		} {
			name := name
			Default.Register(Kernel{
				Name: name,
				Signature: Signature{
					Layout:  tensor.LayoutDense,
					Inputs:  []dtype.DType{paramDType, paramDType},
					Outputs: []dtype.DType{paramDType},
				},
				Locations: []tensor.Location{tensor.Host},
				Run: func(args ...tensor.Tensor) error {
					return runPairLossMixed(args, paramDType, name)
				},
			})
		}

		// Cross-entropy: logits at paramDType, targets Int32, output paramDType.
		Default.Register(Kernel{
			Name: "cross_entropy",
			Signature: Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{paramDType, dtype.Int32},
				Outputs: []dtype.DType{paramDType},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return runCrossEntropyMixed(args, paramDType)
			},
		})
	}
}

func runPairLossMixed(args []tensor.Tensor, kind dtype.DType, name string) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	predLen, err := argLen(args[0], kind)
	if err != nil {
		return err
	}

	targetLen, err := argLen(args[1], kind)
	if err != nil {
		return err
	}

	if predLen != targetLen {
		return tensor.ErrShapeMismatch
	}

	predF32 := BorrowFloat32Buffer(predLen)
	targetF32 := BorrowFloat32Buffer(targetLen)

	defer ReleaseFloat32Buffer(predF32)
	defer ReleaseFloat32Buffer(targetF32)

	if err := widenToF32(args[0], kind, predF32); err != nil {
		return err
	}
	if err := widenToF32(args[1], kind, targetF32); err != nil {
		return err
	}

	result := computeLossF32(predF32, targetF32, name)

	outOne := []float32{result}
	return narrowFromF32(args[2], kind, outOne)
}

func computeLossF32(predictions, targets []float32, name string) float32 {
	var sum float64
	n := float64(len(predictions))

	switch name {
	case "mse_loss":
		for index, value := range predictions {
			delta := float64(value - targets[index])
			sum += delta * delta
		}
		return float32(sum / n)

	case "mae_loss":
		for index, value := range predictions {
			sum += math.Abs(float64(value - targets[index]))
		}
		return float32(sum / n)

	case "huber_loss":
		const delta = float32(1.0)
		for index, value := range predictions {
			diff := value - targets[index]
			absDiff := float32(math.Abs(float64(diff)))

			switch {
			case absDiff <= delta:
				sum += 0.5 * float64(diff) * float64(diff)
			default:
				sum += float64(delta) * (float64(absDiff) - 0.5*float64(delta))
			}
		}
		return float32(sum / n)

	case "binary_cross_entropy":
		const eps = 1e-7
		for index, value := range predictions {
			clamped := math.Max(eps, math.Min(1-eps, float64(value)))
			sum += -float64(targets[index])*math.Log(clamped) -
				(1-float64(targets[index]))*math.Log(1-clamped)
		}
		return float32(sum / n)

	case "kl_divergence":
		const eps = 1e-12
		for index, value := range predictions {
			p := math.Max(eps, float64(value))
			q := math.Max(eps, float64(targets[index]))
			sum += q * math.Log(q/p)
		}
		return float32(sum / n)
	}

	return 0
}

func runCrossEntropyMixed(args []tensor.Tensor, kind dtype.DType) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	logitsLen, err := argLen(args[0], kind)
	if err != nil {
		return err
	}

	targets, err := args[1].Int32Native()
	if err != nil {
		return err
	}

	logitDims := args[0].Shape().Dims()
	if len(logitDims) < 1 {
		return tensor.ErrShapeMismatch
	}

	classes := logitDims[len(logitDims)-1]
	batchSize := logitsLen / classes

	if len(targets) != batchSize {
		return tensor.ErrShapeMismatch
	}

	logitsF32 := BorrowFloat32Buffer(logitsLen)
	defer ReleaseFloat32Buffer(logitsF32)

	if err := widenToF32(args[0], kind, logitsF32); err != nil {
		return err
	}

	var sum float64

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		row := logitsF32[batchIndex*classes : (batchIndex+1)*classes]
		maxLogit := row[0]

		for _, candidate := range row[1:] {
			if candidate > maxLogit {
				maxLogit = candidate
			}
		}

		var denominator float64

		for _, candidate := range row {
			denominator += math.Exp(float64(candidate - maxLogit))
		}

		targetClass := int(targets[batchIndex])

		if targetClass < 0 || targetClass >= classes {
			return tensor.ErrShapeMismatch
		}

		logProb := float64(row[targetClass]-maxLogit) - math.Log(denominator)
		sum += -logProb
	}

	result := float32(sum / float64(batchSize))
	return narrowFromF32(args[2], kind, []float32{result})
}
