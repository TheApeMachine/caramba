/*
Package service provides the Agent-to-Agent (A2A) communication service, enabling
seamless interaction between AI agents through a REST API and Server-Sent Events.
*/

package service

import (
	"bufio"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/caramba/pkg/agent/handlers"
	"github.com/theapemachine/caramba/pkg/catalog"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/stores/s3"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"golang.org/x/crypto/acme/autocert"
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
	app             *fiber.App
	certManager     *autocert.Manager
	card            *agent.Card
	streams         map[string][]*taskStream
	streamMutex     sync.RWMutex
	notificationMgr *task.NotificationManager
	middleware      *Middleware
	catalog         *catalog.Catalog
	taskStore       task.TaskStore
	s3Repo          *s3.Repository
	bucketName      string
	llmProvider     provider.ProviderType
	toolRegistry    *tools.Registry
}

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
func NewA2A() *A2A {
	s3Conn := &s3.Conn{}
	bucketName := tweaker.Value[string]("settings.s3.bucketName")
	if bucketName == "" {
		bucketName = "your-default-bucket"
	}
	s3Repo := s3.NewRepository(s3Conn, bucketName)
	taskStore := s3.NewS3TaskStore(s3Repo)

	llmProvider := provider.NewOpenAIProvider()

	// Initialize Tool Registry and Register Tools
	toolRegistry := tools.NewRegistry()
	// Make tool registration configurable or dynamic
	ghTool := tools.NewGithubTool()
	for _, t := range ghTool.Tools {
		toolRegistry.Register(t)
	}
	// Register other tools as needed
	// browserTool := tools.NewBrowserTool()
	// toolRegistry.Register(browserTool)
	// memoryTool := tools.NewMemoryTool()
	// toolRegistry.Register(memoryTool)
	// ... etc

	app := fiber.New(fiber.Config{
		JSONEncoder:       types.SimdMarshalJSON,
		JSONDecoder:       types.SimdUnmarshalJSON,
		ServerHeader:      "Caramba",
		AppName:           "Caramba",
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		StreamRequestBody: true,
		StructValidator:   task.NewGenericValidator(),
	})
	middleware := NewMiddleware(app)
	middleware.Register()

	return &A2A{
		app:        app,
		middleware: middleware,
		certManager: &autocert.Manager{
			Prompt: autocert.AcceptTOS,
			HostPolicy: autocert.HostWhitelist(
				tweaker.Value[string]("settings.domain"),
			),
			Cache: autocert.DirCache("./certs"),
		},
		card:            agent.FromConfig("default"),
		streams:         make(map[string][]*taskStream),
		notificationMgr: task.NewNotificationManager(),
		catalog:         catalog.NewCatalog(),
		taskStore:       taskStore,
		s3Repo:          s3Repo,
		bucketName:      bucketName,
		llmProvider:     llmProvider,
		toolRegistry:    toolRegistry,
	}
}

/*
RegisterRoutes sets up all HTTP endpoints for the A2A service, including
health checks, agent catalog, JSON-RPC endpoints, and SSE streams.
*/
func (srv *A2A) RegisterRoutes() {
	// Root handler
	srv.app.Get("/", func(ctx fiber.Ctx) error {
		return ctx.SendString("OK")
	})

	// To request the agents catalog.
	srv.app.Get("/.well-known/agents/catalog", func(ctx fiber.Ctx) error {
		return ctx.JSON(srv.card)
	})

	// To request an agent card.
	srv.app.Get("/.well-known/agents/:agent", func(ctx fiber.Ctx) error {
		agentName := ctx.Params("agent")
		agent := srv.catalog.GetAgent(agentName)
		if agent == nil {
			return ctx.Status(fiber.StatusNotFound).SendString("Agent not found")
		}
		return ctx.JSON(agent)
	})

	// A2A JSON-RPC endpoint
	srv.app.Post("/rpc", func(ctx fiber.Ctx) error {
		var req types.JSONRPC
		if err := ctx.Bind().Body(&req); err != nil {
			parseErr := errnie.New(
				errnie.WithMessage(fmt.Sprintf("Parse error: %s", err.Error())),
				errnie.WithType(errnie.InvalidInputError),
				errnie.WithStatus(errnie.BadRequestStatus),
				errnie.WithError(err),
			)

			return ctx.Status(parseErr.Status()).JSON(types.JSONRPCResponse{
				Version: "2.0",
				Error: &types.JSONRPCError{
					Code:    -32700,
					Message: "Parse error",
					Data:    err.Error(),
				},
				ID: nil,
			})
		}

		// Ensure it's a valid JSON-RPC 2.0 request
		if req.Version != "2.0" {
			validationErr := errnie.New(
				errnie.WithMessage("Invalid Request: Expected JSON-RPC 2.0"),
				errnie.WithType(errnie.ValidationError),
				errnie.WithStatus(errnie.BadRequestStatus),
			)

			return ctx.Status(validationErr.Status()).JSON(types.JSONRPCResponse{
				Version: "2.0",
				Error: &types.JSONRPCError{
					Code:    -32600,
					Message: "Invalid Request",
					Data:    "Expected JSON-RPC 2.0",
				},
				ID: req.ID,
			})
		}

		// Process the request based on method
		var result any
		var err *task.TaskRequestError

		switch req.Method {
		case "tasks/send":
			result, err = handlers.HandleTaskSend(srv.taskStore, srv.llmProvider, srv.toolRegistry, srv, req.Params)
		case "tasks/get":
			result, err = handlers.HandleTaskGet(srv.taskStore, req.Params)
		case "tasks/cancel":
			result, err = handlers.HandleTaskCancel(srv.taskStore, req.Params)
		case "tasks/pushNotification/set":
			result, err = handlers.HandleTaskSetPushNotification(srv.taskStore, req.Params)
		case "tasks/pushNotification/get":
			result, err = handlers.HandleTaskGetPushNotification(srv.taskStore, req.Params)
		case "tasks/sendSubscribe":
			handlers.NewSendSubscriberHandler(
				ctx, srv.taskStore, srv.llmProvider, srv.toolRegistry,
			).HandleRequest(
				req.Params, req.ID,
			)
		case "tasks/resubscribe":
			result, err = handlers.NewTaskResubscribeHandler(
				ctx, srv.taskStore,
			).HandleRequest(req.Params)
		default:
			// Method not found error according to JSON-RPC spec
			err = &task.TaskRequestError{
				Code:    -32601, // Method not found
				Message: "Method not found",
				Data:    fmt.Sprintf("Method '%s' not supported", req.Method),
			}
		}

		if err != nil {
			return ctx.JSON(types.JSONRPCResponse{
				Version: "2.0",
				Error:   convertToJSONRPCError(err),
				ID:      req.ID,
			})
		}

		return ctx.JSON(types.JSONRPCResponse{
			Version: "2.0",
			Result:  result,
			ID:      req.ID,
		})
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

	// Send push notification if appropriate
	// Check if the update contains a status
	if statusUpdate, ok := update.(map[string]any); ok {
		if status, hasStatus := statusUpdate["status"]; hasStatus {
			if typedStatus, ok := status.(task.TaskStatus); ok && srv.card.Capabilities.PushNotifications {
				srv.notificationMgr.SendTaskStatusUpdate(
					taskID,
					typedStatus,
					typedStatus.State == task.TaskStateCompleted ||
						typedStatus.State == task.TaskStateFailed ||
						typedStatus.State == task.TaskStateCanceled,
					nil,
				)
			}
		}
	}
}

/*
Listen starts the A2A service on the specified address, registering all routes
and configuring TLS using the certificate manager.
*/
func (srv *A2A) Listen(addr string) error {
	// Register routes before starting the server
	srv.RegisterRoutes()

	return srv.app.Listen(addr, fiber.ListenConfig{
		AutoCertManager: srv.certManager,
	})
}

/*
convertToJSONRPCError transforms a TaskRequestError into a JSONRPCError for
consistent error handling across the API.
*/
func convertToJSONRPCError(err *task.TaskRequestError) *types.JSONRPCError {
	return &types.JSONRPCError{
		Code:    err.Code,
		Message: err.Message,
		Data:    err.Data,
	}
}

// A2A implements StreamUpdateSender implicitly because it has the SendTaskUpdate method.
