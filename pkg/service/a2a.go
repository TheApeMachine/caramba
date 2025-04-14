/*
Package service provides the Agent-to-Agent (A2A) communication service, enabling
seamless interaction between AI agents through a REST API and Server-Sent Events.
*/

package service

import (
	"bufio"
	"bytes"
	"context"
	"log"
	"sync"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
StreamUpdateSender defines the interface for components that can send updates
to task streams. This enables decoupled communication between task processors
and stream managers.
*/
type StreamUpdateSender interface {
	SendTaskUpdate(taskID string, update any)
}

/*
A2A implements the Agent-to-Agent communication service, providing a REST API
for task management and real-time updates through Server-Sent Events. It supports
TLS, rate limiting, authentication, and push notifications.

Example:

	a2a := NewA2A()
	if err := a2a.Listen(":8080"); err != nil {
	    log.Fatal(err)
	}
*/
type A2A struct {
	app         *fiber.App
	agent       agent.Agent
	streams     map[string][]*taskStream
	streamMutex sync.RWMutex
	middleware  *Middleware
}

type A2AOption func(*A2A)

/*
taskStream represents a single client connection for receiving task updates.
It manages message delivery and connection lifecycle.
*/
type taskStream struct {
	taskID   string
	messages chan any
	done     chan struct{}
}

/*
NewA2A creates a new Agent-to-Agent service with configured S3 storage,
LLM provider, and tool registry. It initializes the Fiber application with
appropriate middleware and settings.
*/
func NewA2A(opts ...A2AOption) *A2A {
	a2a := &A2A{
		app: fiber.New(fiber.Config{
			ServerHeader: "Caramba A2A Service",
			ReadTimeout:  10 * time.Second,
			WriteTimeout: 10 * time.Second,
		}),
		streams: make(map[string][]*taskStream),
	}

	for _, opt := range opts {
		opt(a2a)
	}

	return a2a
}

/*
RegisterRoutes sets up all HTTP endpoints for the A2A service, including
health checks, agent catalog, and SSE streams.
*/
func (srv *A2A) RegisterRoutes() {
	ok := bytes.NewReader([]byte("OK"))
	okSize := ok.Len()

	// To request the agents catalog.
	srv.app.Get("/.well-known/agent.json", func(ctx fiber.Ctx) error {
		return ctx.JSON(srv.agent.Card())
	})

	// SSE streaming endpoint for task updates
	srv.app.Get("/task/:id/stream", func(ctx fiber.Ctx) error {
		// Use SendStreamWriter to handle the SSE
		sseHandler := NewSSE(srv)
		handler := sseHandler.Handler(ctx)

		return ctx.SendStreamWriter(func(w *bufio.Writer) {
			if err := handler(w); err != nil {
				// Log the error since we can't return it
				log.Printf("Error in SSE stream: %v", err)
			}
		})
	})

	srv.app.Use([]string{
		"/",
		"/ping",
		"/health",
		"/healthz",
	}, func(ctx fiber.Ctx) error {
		ctx.Set("Content-Type", "text/plain")
		return ctx.Status(fiber.StatusOK).SendStream(ok, okSize)
	})
}

/*
SendTaskUpdate delivers updates to all connected clients for a specific task
and triggers push notifications when appropriate. It handles concurrent access
to streams safely using a read lock.
*/
func (srv *A2A) SendTaskUpdate(taskID string, update any) {
	// Send SSE update
	srv.streamMutex.RLock()
	streams, exists := srv.streams[taskID]
	if exists {
		// Send to all streams for this task
		for _, stream := range streams {
			select {
			case stream.messages <- update:
				// Message sent successfully
			case <-stream.done:
				// Stream is done, skip
			default:
				// Channel buffer is full, log warning
				log.Printf("Warning: stream buffer full for task %s", taskID)
			}
		}
	}
	srv.streamMutex.RUnlock()
}

/*
Listen starts the A2A service on the specified address, registering all routes
and configuring TLS using the certificate manager.
*/
func (srv *A2A) Listen(addr string) error {
	srv.RegisterRoutes()

	go func() {
		if err := srv.StartRPCServer(context.Background(), ":3211"); err != nil {
			errnie.New(errnie.WithError(err))
		}
	}()

	return srv.app.Listen(":"+addr, fiber.ListenConfig{})
}

func WithName(name string) A2AOption {
	return func(a2a *A2A) {
		a2a.app.Name(name)
		a2a.agent = agent.NewBuilder(
			agent.WithCard(agent.FromConfig(name)),
		)
	}
}

func WithMiddleware(middleware *Middleware) A2AOption {
	return func(a2a *A2A) {
		a2a.middleware = middleware
		a2a.middleware.Register(a2a.app)
	}
}

func WithAgent(agent agent.Agent) A2AOption {
	return func(a2a *A2A) {
		a2a.agent = agent
	}
}
