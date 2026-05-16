package shape

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
UpsampleNearest2D expands an NCHW image tensor by integer nearest-neighbor
scale factors.
*/
type UpsampleNearest2D struct{}

func NewUpsampleNearest2D(scale ...int) *UpsampleNearest2D {
	return &UpsampleNearest2D{}
}

func (upsample *UpsampleNearest2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("shape.upsample_nearest2d"); err != nil {
		return nil, err
	}

	batch, channels, height, width, scaleH, scaleW, err := upsampleNearest2DLayout(
		stateDict,
	)

	if err != nil {
		return nil, err
	}

	outputLength, err := upsampleNearest2DOutputLength(
		batch, channels, height, width, scaleH, scaleW,
	)

	if err != nil {
		return nil, err
	}

	stateDict.EnsureOperationOutLen(outputLength)
	upsampleNearest2DKernel(
		stateDict.Out,
		stateDict.Inputs[0],
		batch,
		channels,
		height,
		width,
		scaleH,
		scaleW,
	)

	return stateDict, nil
}

func upsampleNearest2DLayout(
	stateDict *state.Dict,
) (int, int, int, int, int, int, error) {
	shape := stateDict.OperationShape()

	if len(shape) != 4 {
		return 0, 0, 0, 0, 0, 0, fmt.Errorf(
			"shape.upsample_nearest2d: expected NCHW rank 4, got %d",
			len(shape),
		)
	}

	batch, channels, height, width := shape[0], shape[1], shape[2], shape[3]

	if batch <= 0 || channels <= 0 || height <= 0 || width <= 0 {
		return 0, 0, 0, 0, 0, 0, fmt.Errorf(
			"shape.upsample_nearest2d: NCHW dimensions must be positive",
		)
	}

	expectedInput, err := checkedMulInts(batch, channels, height, width)

	if err != nil {
		return 0, 0, 0, 0, 0, 0, err
	}

	if len(stateDict.Inputs[0]) != expectedInput {
		return 0, 0, 0, 0, 0, 0, fmt.Errorf(
			"shape.upsample_nearest2d: input length %d does not match NCHW size %d",
			len(stateDict.Inputs[0]),
			expectedInput,
		)
	}

	scaleH := stateDict.ScaleH
	scaleW := stateDict.ScaleW

	if scaleH == 0 && stateDict.OutH > 0 {
		if stateDict.OutH%height != 0 {
			return 0, 0, 0, 0, 0, 0, fmt.Errorf(
				"shape.upsample_nearest2d: out_h %d is not divisible by height %d",
				stateDict.OutH,
				height,
			)
		}

		scaleH = stateDict.OutH / height
	}

	if scaleW == 0 && stateDict.OutW > 0 {
		if stateDict.OutW%width != 0 {
			return 0, 0, 0, 0, 0, 0, fmt.Errorf(
				"shape.upsample_nearest2d: out_w %d is not divisible by width %d",
				stateDict.OutW,
				width,
			)
		}

		scaleW = stateDict.OutW / width
	}

	if scaleH <= 0 || scaleW <= 0 {
		return 0, 0, 0, 0, 0, 0, fmt.Errorf(
			"shape.upsample_nearest2d: scale_h and scale_w must be positive",
		)
	}

	return batch, channels, height, width, scaleH, scaleW, nil
}

func upsampleNearest2DOutputLength(
	batch int,
	channels int,
	height int,
	width int,
	scaleH int,
	scaleW int,
) (int, error) {
	return checkedMulInts(batch, channels, height, scaleH, width, scaleW)
}

func checkedMulInts(values ...int) (int, error) {
	product := 1

	for _, value := range values {
		if value <= 0 {
			return 0, fmt.Errorf("shape.upsample_nearest2d: dimension must be positive")
		}

		if product > math.MaxInt/value {
			return 0, fmt.Errorf("shape.upsample_nearest2d: shape product overflows int")
		}

		product *= value
	}

	return product, nil
}
