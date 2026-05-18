package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Loss kernels. Each produces a scalar output (length-1 float32) that
represents the mean loss across the batch. Args order:
(predictions, targets, output).

For cross-entropy / NLL, the predictions are treated as logits and
the targets as int32 class indices.
*/

func init() {
	Default.Register(Kernel{
		Name: "mse_loss",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMSELoss,
	})

	Default.Register(Kernel{
		Name: "mae_loss",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMAELoss,
	})

	Default.Register(Kernel{
		Name: "huber_loss",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runHuberLoss,
	})

	Default.Register(Kernel{
		Name: "binary_cross_entropy",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runBinaryCrossEntropy,
	})

	Default.Register(Kernel{
		Name: "cross_entropy",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Int32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runCrossEntropy,
	})

	Default.Register(Kernel{
		Name: "kl_divergence",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runKLDivergence,
	})
}

func loadLossPair(args []tensor.Tensor) ([]float32, []float32, []float32, error) {
	predictions, err := args[0].Float32Native()

	if err != nil {
		return nil, nil, nil, err
	}

	targets, err := args[1].Float32Native()

	if err != nil {
		return nil, nil, nil, err
	}

	out, err := args[2].Float32Native()

	if err != nil {
		return nil, nil, nil, err
	}

	if len(predictions) != len(targets) || len(out) < 1 {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	return predictions, targets, out, nil
}

func runMSELoss(args ...tensor.Tensor) error {
	predictions, targets, out, err := loadLossPair(args)

	if err != nil {
		return err
	}

	var sum float64

	for index, value := range predictions {
		delta := float64(value - targets[index])
		sum += delta * delta
	}

	out[0] = float32(sum / float64(len(predictions)))
	return nil
}

func runMAELoss(args ...tensor.Tensor) error {
	predictions, targets, out, err := loadLossPair(args)

	if err != nil {
		return err
	}

	var sum float64

	for index, value := range predictions {
		sum += math.Abs(float64(value - targets[index]))
	}

	out[0] = float32(sum / float64(len(predictions)))
	return nil
}

func runHuberLoss(args ...tensor.Tensor) error {
	predictions, targets, out, err := loadLossPair(args)

	if err != nil {
		return err
	}

	const delta = float32(1.0)
	var sum float64

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

	out[0] = float32(sum / float64(len(predictions)))
	return nil
}

func runBinaryCrossEntropy(args ...tensor.Tensor) error {
	predictions, targets, out, err := loadLossPair(args)

	if err != nil {
		return err
	}

	var sum float64
	const eps = 1e-7

	for index, value := range predictions {
		clamped := math.Max(eps, math.Min(1-eps, float64(value)))
		sum += -float64(targets[index])*math.Log(clamped) -
			(1-float64(targets[index]))*math.Log(1-clamped)
	}

	out[0] = float32(sum / float64(len(predictions)))
	return nil
}

func runCrossEntropy(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	logits, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	targets, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	logitDims := args[0].Shape().Dims()

	if len(logitDims) < 1 || len(out) < 1 {
		return tensor.ErrShapeMismatch
	}

	classes := logitDims[len(logitDims)-1]
	batchSize := len(logits) / classes

	if len(targets) != batchSize {
		return tensor.ErrShapeMismatch
	}

	var sum float64

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		row := logits[batchIndex*classes : (batchIndex+1)*classes]
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

	out[0] = float32(sum / float64(batchSize))
	return nil
}

func runKLDivergence(args ...tensor.Tensor) error {
	predictions, targets, out, err := loadLossPair(args)

	if err != nil {
		return err
	}

	var sum float64
	const eps = 1e-12

	for index, value := range predictions {
		p := math.Max(eps, float64(value))
		q := math.Max(eps, float64(targets[index]))
		sum += q * math.Log(q/p)
	}

	out[0] = float32(sum / float64(len(predictions)))
	return nil
}
