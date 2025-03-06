package workflow

import (
	"encoding/json"
	"io"
)

// TransformMiddleware creates a component that transforms data between two types
func TransformMiddleware[In any, Out any](
	stream io.ReadWriteCloser,
	transformFn func(In) (Out, error),
) io.ReadWriteCloser {
	return &transformComponent[In, Out]{
		stream:      stream,
		transformFn: transformFn,
	}
}

type transformComponent[In any, Out any] struct {
	stream      io.ReadWriteCloser
	transformFn func(In) (Out, error)
}

func (tc *transformComponent[In, Out]) Read(p []byte) (int, error) {
	var buffer [4096]byte
	n, err := tc.stream.Read(buffer[:])
	if err != nil && err != io.EOF {
		return 0, err
	}

	var input In
	if err := json.Unmarshal(buffer[:n], &input); err != nil {
		return 0, err
	}

	output, err := tc.transformFn(input)
	if err != nil {
		return 0, err
	}

	data, err := json.Marshal(output)
	if err != nil {
		return 0, err
	}

	return copy(p, data), nil
}

func (tc *transformComponent[In, Out]) Write(p []byte) (int, error) {
	return tc.stream.Write(p)
}

func (tc *transformComponent[In, Out]) Close() error {
	return tc.stream.Close()
}
