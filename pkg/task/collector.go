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
		response: NewTaskResponse(WithResponseTask(NewTask())),
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

	// Clear the buffer since we've processed all complete JSON objects
	tc.buffer.Reset()

	return n, nil
}
