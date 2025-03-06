package workflow

import (
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func NewCodecStream(codec *JSONCodec) *CodecStream {
	return &CodecStream{codec: codec}
}

func (cs *CodecStream) Read(p []byte) (int, error) {
	errnie.Debug("Reading from codec stream")

	var v any
	if err := cs.codec.Read(&v); err != nil {
		errnie.Error("Failed to read from codec stream", "error", err)
		return 0, err
	}

	data, err := json.Marshal(v)
	if err != nil {
		errnie.Error("Failed to marshal data", "error", err)
		return 0, err
	}

	if len(p) < len(data) {
		errnie.Error("Short buffer")
		return 0, io.ErrShortBuffer
	}

	return copy(p, data), nil
}

func (cs *CodecStream) Write(p []byte) (int, error) {
	errnie.Debug("Writing to codec stream", "len", len(p))
	errnie.Debug(string(p))

	var v any
	if err := json.Unmarshal(p, &v); err != nil {
		errnie.Error("Failed to unmarshal data", "error", err)
		return 0, err
	}

	if err := cs.codec.Write(v); err != nil {
		errnie.Error("Failed to write to codec", "error", err)
		return 0, err
	}

	return len(p), nil
}

func (cs *CodecStream) Close() error {
	return cs.codec.Close()
}
