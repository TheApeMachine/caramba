/*
Package service provides server-sent events (SSE) functionality for real-time updates.
It enables clients to receive continuous updates from the server without maintaining
persistent WebSocket connections.
*/

package service

import (
	"bufio"
	"encoding/json"
	"fmt"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/task"
)

/*
SSE represents a Server-Sent Events service that manages real-time communication
between the server and clients. It's designed to be lightweight and efficient,
using HTTP/1.1 chunked transfer encoding for streaming updates.

Example:

	sse := NewSSE(a2aService)
	app.Get("/events/:id", func(c fiber.Ctx) error {
	    return sse.Handler(c)(bufio.NewWriter(c.Response().BodyWriter()))
	})
*/
type SSE struct {
	app     *fiber.App
	service *A2A
}

/*
NewSSE creates a new SSE service instance. It's the primary entry point for
setting up server-sent events functionality within your application.
*/
func NewSSE(service *A2A) *SSE {
	return &SSE{
		app:     service.app,
		service: service,
	}
}

/*
Handler returns a function that manages the SSE connection lifecycle. It handles
client connections, message streaming, and connection cleanup.

The handler is designed to be used with Fiber's routing system, providing a
seamless integration with the web framework.
*/
func (sse *SSE) Handler(ctx fiber.Ctx) func(*bufio.Writer) error {
	return func(writer *bufio.Writer) error {
		taskID := ctx.Params("id")

		if taskID == "" {
			validationErr := errnie.New(
				errnie.WithMessage("Task ID is required"),
				errnie.WithType(errnie.ValidationError),
				errnie.WithStatus(errnie.BadRequestStatus),
			)

			return ctx.Status(
				validationErr.Status(),
			).SendString("Task ID is required")
		}

		sse.setupSSEHeaders(ctx)
		stream := sse.registerStream(taskID)

		sse.sendInitialMessage(writer, taskID)
		return sse.handleEventLoop(ctx, writer, stream, taskID)
	}
}

/*
setupSSEHeaders configures the necessary HTTP headers for SSE connections.
These headers are crucial for maintaining the connection and ensuring proper
client-side handling of the event stream.
*/
func (sse *SSE) setupSSEHeaders(ctx fiber.Ctx) {
	ctx.Set("Content-Type", "text/event-stream")
	ctx.Set("Cache-Control", "no-cache")
	ctx.Set("Connection", "keep-alive")
	ctx.Set("Transfer-Encoding", "chunked")
}

/*
registerStream creates a new event stream for a specific task. It manages the
concurrent access to streams using mutex locks to ensure thread safety.
*/
func (sse *SSE) registerStream(taskID string) *taskStream {
	stream := &taskStream{
		taskID:   taskID,
		messages: make(chan any, 100),
		done:     make(chan struct{}),
	}

	sse.service.streamMutex.Lock()

	if _, exists := sse.service.streams[taskID]; !exists {
		sse.service.streams[taskID] = make([]*taskStream, 0)
	}

	sse.service.streams[taskID] = append(sse.service.streams[taskID], stream)
	sse.service.streamMutex.Unlock()

	return stream
}

/*
sendInitialMessage establishes the SSE connection by sending an initial
message to the client. This helps clients confirm the connection is active
and identify the specific task they're subscribed to.
*/
func (sse *SSE) sendInitialMessage(writer *bufio.Writer, taskID string) {
	fmt.Fprintf(writer, "event: connected\n")
	fmt.Fprintf(writer, "data: {\"taskId\": \"%s\"}\n\n", taskID)
	writer.Flush()
}

/*
handleEventLoop manages the core event processing loop for an SSE connection.
It handles three main aspects:
1. Client disconnection detection
2. Message processing and delivery
3. Connection keep-alive through periodic pings

The loop continues until the client disconnects or an error occurs.
*/
func (sse *SSE) handleEventLoop(ctx fiber.Ctx, writer *bufio.Writer, stream *taskStream, taskID string) error {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	defer func() {
		close(stream.done)
		sse.removeStream(taskID, stream)
	}()

	for {
		select {
		case <-ctx.Context().Done():
			errnie.Info("client disconnected", "task", taskID)
			return nil

		case msg, ok := <-stream.messages:
			if !ok {
				return nil // Channel was closed
			}

			if err := sse.sendMessage(writer, msg); err != nil {
				return err
			}

		case <-ticker.C:
			if err := sse.sendPing(writer); err != nil {
				return err
			}
		}
	}
}

/*
sendMessage serializes and delivers messages to connected clients. It handles
JSON serialization and ensures proper SSE message formatting with event types
and data fields.
*/
func (sse *SSE) sendMessage(writer *bufio.Writer, msg any) error {
	// Serialize the message to JSON
	data, err := json.Marshal(msg)
	if err != nil {
		errnie.New(errnie.WithError(err))
		return nil
	}

	// Determine event type based on message type
	var eventType string
	switch msg.(type) {
	case task.TaskStatusUpdateEvent:
		eventType = "task.status"
	case task.TaskArtifactUpdateEvent:
		eventType = "task.artifact"
	default:
		eventType = "update" // Fallback for other message types
	}

	// Write event data with correct event type
	fmt.Fprintf(writer, "event: %s\n", eventType)
	fmt.Fprintf(writer, "data: %s\n\n", string(data))

	if err := writer.Flush(); err != nil {
		errnie.New(errnie.WithError(err))
		return err
	}

	return nil
}

/*
sendPing maintains the SSE connection by sending periodic keep-alive messages.
This prevents proxy servers and load balancers from closing idle connections.
*/
func (sse *SSE) sendPing(writer *bufio.Writer) error {
	fmt.Fprintf(writer, "event: ping\n")
	fmt.Fprintf(writer, "data: {\"timestamp\": %d}\n\n", time.Now().Unix())

	if err := writer.Flush(); err != nil {
		errnie.New(errnie.WithError(err))
		return err
	}

	return nil
}

/*
removeStream cleans up resources when a client disconnects. It safely removes
the stream from the service's stream registry and handles cleanup of empty
stream lists.
*/
func (sse *SSE) removeStream(taskID string, stream *taskStream) {
	sse.service.streamMutex.Lock()
	defer sse.service.streamMutex.Unlock()

	streams, exists := sse.service.streams[taskID]

	if !exists {
		return
	}

	for i, st := range streams {
		if st == stream {
			sse.service.streams[taskID] = append(streams[:i], streams[i+1:]...)
			break
		}
	}

	if len(sse.service.streams[taskID]) == 0 {
		delete(sse.service.streams, taskID)
	}
}
