package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func init() {
	registerLayerNormDType(dtype.Float16, runLayerNormFloat16)
	registerLayerNormDType(dtype.BFloat16, runLayerNormBFloat16)
	registerRMSNormDType(dtype.Float16, runRMSNormFloat16)
	registerRMSNormDType(dtype.BFloat16, runRMSNormBFloat16)
}

func registerLayerNormDType(storageDType dtype.DType, run func(...tensor.Tensor) error) {
	Default.Register(Kernel{
		Name: "layernorm",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType, storageDType, storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       run,
	})
}

func registerRMSNormDType(storageDType dtype.DType, run func(...tensor.Tensor) error) {
	Default.Register(Kernel{
		Name: "rmsnorm",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       run,
	})
}

func runLayerNormFloat16(args ...tensor.Tensor) error {
	input, scale, bias, out, lastDim, err := layerNormFloat16Views(args)
	if err != nil {
		return err
	}

	rows := len(input) / lastDim

	for rowIndex := range rows {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := out[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyLayerNormFloat16Row(row, outRow, scale, bias)
	}

	return nil
}

func runLayerNormBFloat16(args ...tensor.Tensor) error {
	input, scale, bias, out, lastDim, err := layerNormBFloat16Views(args)
	if err != nil {
		return err
	}

	rows := len(input) / lastDim

	for rowIndex := range rows {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := out[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyLayerNormBFloat16Row(row, outRow, scale, bias)
	}

	return nil
}

func runRMSNormFloat16(args ...tensor.Tensor) error {
	input, scale, out, lastDim, err := rmsNormFloat16Views(args)
	if err != nil {
		return err
	}

	rows := len(input) / lastDim

	// Bulk widen via NEON; run vectorized f32 RMSNorm; bulk narrow.
	inputF32 := borrowFloat32Buffer(len(input))
	scaleF32 := borrowFloat32Buffer(lastDim)
	outF32 := borrowFloat32Buffer(len(input))

	defer releaseFloat32Buffer(inputF32)
	defer releaseFloat32Buffer(scaleF32)
	defer releaseFloat32Buffer(outF32)

	float16BulkToFloat32(inputF32, input)
	float16BulkToFloat32(scaleF32, scale)

	for rowIndex := range rows {
		row := inputF32[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := outF32[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyRMSRow(row, outRow, scaleF32)
	}

	float32BulkToFloat16(out, outF32)
	return nil
}

func runRMSNormBFloat16(args ...tensor.Tensor) error {
	input, scale, out, lastDim, err := rmsNormBFloat16Views(args)
	if err != nil {
		return err
	}

	rows := len(input) / lastDim

	inputF32 := borrowFloat32Buffer(len(input))
	scaleF32 := borrowFloat32Buffer(lastDim)
	outF32 := borrowFloat32Buffer(len(input))

	defer releaseFloat32Buffer(inputF32)
	defer releaseFloat32Buffer(scaleF32)
	defer releaseFloat32Buffer(outF32)

	bfloat16BulkToFloat32(inputF32, input)
	bfloat16BulkToFloat32(scaleF32, scale)

	for rowIndex := range rows {
		row := inputF32[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := outF32[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyRMSRow(row, outRow, scaleF32)
	}

	float32BulkToBFloat16(out, outF32)
	return nil
}

func layerNormFloat16Views(
	args []tensor.Tensor,
) (input, scale, bias, out []dtype.F16, lastDim int, err error) {
	if err := requireLayerNormArgCount(args); err != nil {
		return nil, nil, nil, nil, 0, err
	}

	lastDim, err = normLastDim(args[0], args[3])
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	input, err = args[0].Float16Native()
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	scale, err = args[1].Float16Native()
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	bias, err = args[2].Float16Native()
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	out, err = args[3].Float16Native()
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	return input, scale, bias, out, lastDim, requireNormScaleBias(scale, bias, lastDim)
}

func layerNormBFloat16Views(
	args []tensor.Tensor,
) (input, scale, bias, out []dtype.BF16, lastDim int, err error) {
	if err := requireLayerNormArgCount(args); err != nil {
		return nil, nil, nil, nil, 0, err
	}

	lastDim, err = normLastDim(args[0], args[3])
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	input, err = args[0].BFloat16Native()
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	scale, err = args[1].BFloat16Native()
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	bias, err = args[2].BFloat16Native()
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	out, err = args[3].BFloat16Native()
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	return input, scale, bias, out, lastDim, requireNormScaleBias(scale, bias, lastDim)
}

func rmsNormFloat16Views(
	args []tensor.Tensor,
) (input, scale, out []dtype.F16, lastDim int, err error) {
	if err := requireRMSNormArgCount(args); err != nil {
		return nil, nil, nil, 0, err
	}

	lastDim, err = normLastDim(args[0], args[2])
	if err != nil {
		return nil, nil, nil, 0, err
	}

	input, err = args[0].Float16Native()
	if err != nil {
		return nil, nil, nil, 0, err
	}

	scale, err = args[1].Float16Native()
	if err != nil {
		return nil, nil, nil, 0, err
	}

	out, err = args[2].Float16Native()
	if err != nil {
		return nil, nil, nil, 0, err
	}

	return input, scale, out, lastDim, requireNormScale(scale, lastDim)
}

func rmsNormBFloat16Views(
	args []tensor.Tensor,
) (input, scale, out []dtype.BF16, lastDim int, err error) {
	if err := requireRMSNormArgCount(args); err != nil {
		return nil, nil, nil, 0, err
	}

	lastDim, err = normLastDim(args[0], args[2])
	if err != nil {
		return nil, nil, nil, 0, err
	}

	input, err = args[0].BFloat16Native()
	if err != nil {
		return nil, nil, nil, 0, err
	}

	scale, err = args[1].BFloat16Native()
	if err != nil {
		return nil, nil, nil, 0, err
	}

	out, err = args[2].BFloat16Native()
	if err != nil {
		return nil, nil, nil, 0, err
	}

	return input, scale, out, lastDim, requireNormScale(scale, lastDim)
}

func requireLayerNormArgCount(args []tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func requireRMSNormArgCount(args []tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func normLastDim(input tensor.Tensor, out tensor.Tensor) (int, error) {
	if !input.Shape().Equal(out.Shape()) {
		return 0, tensor.ErrShapeMismatch
	}

	dims := input.Shape().Dims()
	if len(dims) == 0 {
		return 0, tensor.ErrShapeMismatch
	}

	return dims[len(dims)-1], nil
}

func requireNormScale[T any](scale []T, lastDim int) error {
	if len(scale) != lastDim {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func requireNormScaleBias[T any](scale []T, bias []T, lastDim int) error {
	if len(scale) != lastDim || len(bias) != lastDim {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func applyLayerNormFloat16Row(
	row []dtype.F16,
	outRow []dtype.F16,
	scale []dtype.F16,
	bias []dtype.F16,
) {
	mean, invStdDev := layerNormStatsFloat16(row)

	for index, value := range row {
		normalized := (value.Float32() - mean) * invStdDev
		outRow[index] = dtype.Fromfloat32(normalized*scale[index].Float32() + bias[index].Float32())
	}
}

func applyLayerNormBFloat16Row(
	row []dtype.BF16,
	outRow []dtype.BF16,
	scale []dtype.BF16,
	bias []dtype.BF16,
) {
	mean, invStdDev := layerNormStatsBFloat16(row)

	for index := range row {
		normalized := ((&row[index]).Float32() - mean) * invStdDev
		outRow[index] = dtype.NewBfloat16FromFloat32(
			normalized*(&scale[index]).Float32() + (&bias[index]).Float32(),
		)
	}
}

func applyRMSNormFloat16Row(row []dtype.F16, outRow []dtype.F16, scale []dtype.F16) {
	invRMS := rmsNormScaleFloat16(row)

	for index, value := range row {
		outRow[index] = dtype.Fromfloat32(value.Float32() * invRMS * scale[index].Float32())
	}
}

func applyRMSNormBFloat16Row(row []dtype.BF16, outRow []dtype.BF16, scale []dtype.BF16) {
	invRMS := rmsNormScaleBFloat16(row)

	for index := range row {
		value := (&row[index]).Float32() * invRMS * (&scale[index]).Float32()
		outRow[index] = dtype.NewBfloat16FromFloat32(value)
	}
}

func layerNormStatsFloat16(row []dtype.F16) (float32, float32) {
	var sum float32

	for _, value := range row {
		sum += value.Float32()
	}

	mean := sum / float32(len(row))
	var variance float32

	for _, value := range row {
		delta := value.Float32() - mean
		variance += delta * delta
	}

	return mean, 1 / float32(math.Sqrt(float64(variance/float32(len(row))+layerNormEpsilon)))
}

func layerNormStatsBFloat16(row []dtype.BF16) (float32, float32) {
	var sum float32

	for index := range row {
		sum += (&row[index]).Float32()
	}

	mean := sum / float32(len(row))
	var variance float32

	for index := range row {
		delta := (&row[index]).Float32() - mean
		variance += delta * delta
	}

	return mean, 1 / float32(math.Sqrt(float64(variance/float32(len(row))+layerNormEpsilon)))
}

func rmsNormScaleFloat16(row []dtype.F16) float32 {
	var meanSquare float32

	for _, value := range row {
		f32 := value.Float32()
		meanSquare += f32 * f32
	}

	return 1 / float32(math.Sqrt(float64(meanSquare/float32(len(row))+rmsNormEpsilon)))
}

func rmsNormScaleBFloat16(row []dtype.BF16) float32 {
	var meanSquare float32

	for index := range row {
		f32 := (&row[index]).Float32()
		meanSquare += f32 * f32
	}

	return 1 / float32(math.Sqrt(float64(meanSquare/float32(len(row))+rmsNormEpsilon)))
}
