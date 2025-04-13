package task

import (
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

// TaskCollector streams raw data and attempts to decode the last complete TaskResponse chunk.
type TaskCollector struct {
	task     *TaskRequest
	response *TaskResponse
	stream   io.Writer
}

// NewTaskCollector creates a collector that writes raw data to stream
// and decodes the last complete TaskResponse chunk it receives.
func NewTaskCollector(task *TaskRequest, stream io.Writer) *TaskCollector {
	return &TaskCollector{
		task:     task,
		response: nil,
		stream:   stream,
	}
}

// Response returns the last successfully decoded TaskResponse.
func (tc *TaskCollector) Response() *TaskResponse {
	return tc.response
}

// Write writes the raw byte slice p (assumed to be a complete JSON chunk)
// to the configured stream and attempts to decode it as a TaskResponse,
// updating the stored response if successful.
func (tc *TaskCollector) Write(p []byte) (n int, err error) {
	errnie.Trace("task collector.Write", "p", string(p))

	// Skip processing if byte slice is empty
	if len(p) == 0 {
		errnie.Debug("Received empty byte slice, skipping")
		return 0, nil
	}

	tempResponse := new(TaskResponse)
	if err = json.Unmarshal(p, tempResponse); err != nil {
		errnie.Warn("Error decoding task response", errnie.WithError(err))
		return 0, err
	}

	tc.task.AddResult(tempResponse)

	// Process history if we have any
	if len(tempResponse.Result.History) > 0 {
		for _, result := range tempResponse.Result.History {
			for _, part := range result.Parts {
				if n, err = tc.stream.Write([]byte(part.Text)); err != nil {
					errnie.Warn(
						"error writing to collector's raw stream",
						errnie.WithError(err),
					)
					return n, err
				}
			}
		}
	}

	return n, nil
}
