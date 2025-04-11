package agent

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type SSE struct {
	app     *fiber.App
	service *Service // Reference to the service
}

func NewSSE(service *Service) *SSE {
	return &SSE{
		app:     service.app,
		service: service,
	}
}

func (sse *SSE) Handler(ctx fiber.Ctx) func(*bufio.Writer) error {
	return func(writer *bufio.Writer) error {
		taskID := ctx.Params("id")
		if taskID == "" {
			validationErr := errnie.New(
				errnie.WithMessage("Task ID is required"),
				errnie.WithType(errnie.ValidationError),
				errnie.WithStatus(errnie.BadRequestStatus),
			)

			return ctx.Status(validationErr.Status()).SendString("Task ID is required")
		}

		// Set required SSE headers
		ctx.Set("Content-Type", "text/event-stream")
		ctx.Set("Cache-Control", "no-cache")
		ctx.Set("Connection", "keep-alive")
		ctx.Set("Transfer-Encoding", "chunked")

		// Create a new stream for this task
		stream := &taskStream{
			taskID:   taskID,
			messages: make(chan interface{}, 100),
			done:     make(chan struct{}),
		}

		// Register the stream
		sse.service.streamMutex.Lock()
		if _, exists := sse.service.streams[taskID]; !exists {
			sse.service.streams[taskID] = make([]*taskStream, 0)
		}
		sse.service.streams[taskID] = append(sse.service.streams[taskID], stream)
		sse.service.streamMutex.Unlock()

		// Send initial connection established message
		fmt.Fprintf(writer, "event: connected\n")
		fmt.Fprintf(writer, "data: {\"taskId\": \"%s\"}\n\n", taskID)
		writer.Flush()

		// Keep-alive ticker
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		// Clean up when the client disconnects
		defer func() {
			close(stream.done)
			sse.removeStream(taskID, stream)
		}()

		for {
			select {
			case <-ctx.Context().Done():
				// Client disconnected
				log.Printf("Client disconnected from task %s stream", taskID)
				return nil

			case msg, ok := <-stream.messages:
				if !ok {
					// Channel was closed
					return nil
				}

				// Serialize the message to JSON
				data, err := json.Marshal(msg)
				if err != nil {
					log.Printf("Error marshaling message: %v", err)
					continue
				}

				// Write event data
				fmt.Fprintf(writer, "event: update\n")
				fmt.Fprintf(writer, "data: %s\n\n", string(data))

				if err := writer.Flush(); err != nil {
					log.Printf("Error flushing data: %v", err)
					return err
				}

			case <-ticker.C:
				// Send a keep-alive ping
				fmt.Fprintf(writer, "event: ping\n")
				fmt.Fprintf(writer, "data: {\"timestamp\": %d}\n\n", time.Now().Unix())

				if err := writer.Flush(); err != nil {
					log.Printf("Error flushing ping: %v", err)
					return err
				}
			}
		}
	}
}

// removeStream removes a stream from the streams map
func (sse *SSE) removeStream(taskID string, stream *taskStream) {
	sse.service.streamMutex.Lock()
	defer sse.service.streamMutex.Unlock()

	streams, exists := sse.service.streams[taskID]
	if !exists {
		return
	}

	// Find and remove the stream
	for i, st := range streams {
		if st == stream {
			// Remove the stream from the slice
			sse.service.streams[taskID] = append(streams[:i], streams[i+1:]...)
			break
		}
	}

	// If no more streams for this task, remove the task entry
	if len(sse.service.streams[taskID]) == 0 {
		delete(sse.service.streams, taskID)
	}
}
