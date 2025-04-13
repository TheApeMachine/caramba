package task

import (
	"bytes"
	"encoding/json"
	"io"
	"log"

	"github.com/theapemachine/caramba/pkg/errnie"
)

// TaskCollector streams raw data and attempts to decode the last complete TaskResponse chunk.
type TaskCollector struct {
	response *TaskResponse // Stores the *last* successfully decoded response
	stream   io.Writer     // The stream to write raw data to (e.g., os.Stdout)
}

// NewTaskCollector creates a collector that writes raw data to stream
// and decodes the last complete TaskResponse chunk it receives.
func NewTaskCollector(stream io.Writer) *TaskCollector {
	return &TaskCollector{
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
	// 1. Write raw data to the underlying stream immediately
	n, err = tc.stream.Write(p)
	if err != nil {
		// Log error writing to the raw stream
		errnie.Warn("Error writing to collector's raw stream", errnie.WithError(err))
		// Depending on desired behavior, might want to return error here
	}

	// 2. Attempt to decode the incoming chunk p directly
	tempResponse := new(TaskResponse)
	// Use a decoder for potentially better error handling / streaming json if needed in future
	decoder := json.NewDecoder(bytes.NewReader(p))
	if decodeErr := decoder.Decode(tempResponse); decodeErr == nil {
		// Successfully decoded this chunk, update the stored response
		tc.response = tempResponse
		// Log successful decoding for debugging
		// log.Printf("Collector successfully decoded chunk: %+v", tempResponse)
	} else {
		// Decoding failed. This indicates the chunk wasn't a valid TaskResponse JSON.
		// This might be expected if other JSON structures (like errors) are sent.
		log.Printf("Collector decode failed for chunk: %v\nChunk content: %s", decodeErr, string(p))
		// Do not return an error here, as the raw write might have succeeded,
		// and failure to decode might be acceptable for some message types.
	}

	// Return the number of bytes written to the raw stream and nil error
	// (unless the raw write itself failed critically)
	return n, nil // Return the original write error if it was critical
}
