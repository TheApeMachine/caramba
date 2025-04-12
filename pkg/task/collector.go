package task

import (
	"bytes"
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type TaskCollector struct {
	response *TaskResponse
	buffer   *bytes.Buffer
	stream   io.Writer
	decoder  *json.Decoder
}

func NewTaskCollector(stream io.Writer) *TaskCollector {
	buffer := bytes.NewBuffer([]byte{})

	return &TaskCollector{
		response: NewTaskResponse(NewTask()),
		buffer:   buffer,
		stream:   stream,
		decoder:  json.NewDecoder(buffer),
	}
}

func (tc *TaskCollector) Response() *TaskResponse {
	return tc.response
}

func (tc *TaskCollector) Write(p []byte) (n int, err error) {
	if n, err = tc.buffer.Write(p); err != nil {
		return n, errnie.New(errnie.WithError(err))
	}

	var chunk TaskResponse

	if err = tc.decoder.Decode(&chunk); err != nil {
		return n, errnie.New(errnie.WithError(err))
	}

	for _, message := range chunk.Result.History {
		tc.response.Result.History = append(
			tc.response.Result.History, message,
		)

		for _, part := range message.Parts {
			tc.stream.Write([]byte(part.Text))
		}
	}

	return n, nil
}
