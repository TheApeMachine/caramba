package agent

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"golang.org/x/crypto/acme/autocert"
)

type Service struct {
	app             *fiber.App
	certManager     *autocert.Manager
	card            Card
	streams         map[string][]*taskStream
	streamMutex     sync.RWMutex
	notificationMgr *NotificationManager
	middleware      *Middleware
}

type taskStream struct {
	taskID   string
	messages chan any
	done     chan struct{}
}

func NewService() *Service {
	app := fiber.New(fiber.Config{
		JSONEncoder:       SimdMarshalJSON,
		JSONDecoder:       SimdUnmarshalJSON,
		ServerHeader:      "Caramba",
		AppName:           "Caramba",
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		StreamRequestBody: true,
		StructValidator:   NewGenericValidator(),
	})
	middleware := NewMiddleware(app)
	middleware.Register()

	return &Service{
		app:        app,
		middleware: middleware,
		certManager: &autocert.Manager{
			Prompt: autocert.AcceptTOS,
			HostPolicy: autocert.HostWhitelist(
				tweaker.Value[string]("settings.domain"),
			),
			Cache: autocert.DirCache("./certs"),
		},
		card: Card{
			Name:        tweaker.Value[string]("settings.agent.name"),
			Description: tweaker.Value[string]("settings.agent.description"),
			URL:         tweaker.Value[string]("settings.domain"),
			Provider: Provider{
				Organization: tweaker.Value[string]("settings.agent.provider.organization"),
				URL:          tweaker.Value[string]("settings.agent.provider.url"),
			},
			Version: tweaker.Value[string]("settings.agent.version"),
			Authentication: Authentication{
				Schemes: tweaker.Value[string]("settings.agent.authentication.schemes"),
			},
			DefaultInputModes:  tweaker.Value[[]string]("settings.agent.defaultInputModes"),
			DefaultOutputModes: tweaker.Value[[]string]("settings.agent.defaultOutputModes"),
			Capabilities: Capabilities{
				Streaming:         tweaker.Value[bool]("settings.agent.capabilities.streaming"),
				PushNotifications: tweaker.Value[bool]("settings.agent.capabilities.pushNotifications"),
			},
		},
		streams:         make(map[string][]*taskStream),
		notificationMgr: NewNotificationManager(),
	}
}

func (srv *Service) RegisterRoutes() {
	// Root handler
	srv.app.Get("/", func(ctx fiber.Ctx) error {
		return ctx.SendString("OK")
	})

	// Serve the A2A agent card
	srv.app.Get("/.well-known/ai-agent.json", func(ctx fiber.Ctx) error {
		return ctx.JSON(srv.card)
	})

	// A2A JSON-RPC endpoint
	srv.app.Post("/rpc", func(ctx fiber.Ctx) error {
		var req JSONRPC
		if err := ctx.Bind().Body(&req); err != nil {
			parseErr := errnie.New(
				errnie.WithMessage(fmt.Sprintf("Parse error: %s", err.Error())),
				errnie.WithType(errnie.InvalidInputError),
				errnie.WithStatus(errnie.BadRequestStatus),
				errnie.WithError(err),
			)

			return ctx.Status(parseErr.Status()).JSON(JSONRPCResponse{
				Version: "2.0",
				Error: &JSONRPCError{
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

			return ctx.Status(validationErr.Status()).JSON(JSONRPCResponse{
				Version: "2.0",
				Error: &JSONRPCError{
					Code:    -32600,
					Message: "Invalid Request",
					Data:    "Expected JSON-RPC 2.0",
				},
				ID: req.ID,
			})
		}

		// Process the request based on method
		var result interface{}
		var err *JSONRPCError

		switch req.Method {
		case "task.create":
			result, err = srv.handleTaskCreate(req.Params)
		case "task.get":
			result, err = srv.handleTaskGet(req.Params)
		case "task.send":
			result, err = srv.handleTaskSend(req.Params)
		case "task.setPushNotification":
			result, err = srv.handleTaskSetPushNotification(req.Params)
		case "task.artifact.create":
			result, err = srv.handleTaskArtifactCreate(req.Params)
		case "task.artifact.get":
			result, err = srv.handleTaskArtifactGet(req.Params)
		default:
			methodErr := errnie.New(
				errnie.WithMessage(fmt.Sprintf("Method not found: %s", req.Method)),
				errnie.WithType(errnie.ResourceNotFoundError),
				errnie.WithStatus(errnie.NotFoundStatus),
			)

			err = &JSONRPCError{
				Code:    -32601,
				Message: "Method not found",
				Data:    fmt.Sprintf("Method '%s' not supported", req.Method),
			}

			errnie.Error(methodErr)
		}

		if err != nil {
			return ctx.JSON(JSONRPCResponse{
				Version: "2.0",
				Error:   err,
				ID:      req.ID,
			})
		}

		return ctx.JSON(JSONRPCResponse{
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

// SendTaskUpdate sends an update to all streams for a specific task
// and also sends a push notification if configured
func (srv *Service) SendTaskUpdate(taskID string, update interface{}) {
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
	if statusUpdate, ok := update.(map[string]interface{}); ok {
		if status, hasStatus := statusUpdate["status"]; hasStatus {
			if typedStatus, ok := status.(TaskStatus); ok && srv.card.Capabilities.PushNotifications {
				srv.notificationMgr.SendTaskStatusUpdate(
					taskID,
					typedStatus,
					typedStatus.State == TaskStateCompleted ||
						typedStatus.State == TaskStateFailed ||
						typedStatus.State == TaskStateCanceled,
					nil,
				)
			}
		}
	}
}

// A2A RPC method handlers
func (srv *Service) handleTaskCreate(params json.RawMessage) (interface{}, *JSONRPCError) {
	// TODO: Implement actual task creation
	taskID := fmt.Sprintf("task-%d", time.Now().UnixNano())

	// Create task with initial status
	status := TaskStatus{
		State:     TaskStateSubmitted,
		Timestamp: time.Now().Format(time.RFC3339),
	}

	// Send SSE update
	srv.SendTaskUpdate(taskID, map[string]interface{}{
		"status": status,
	})

	return map[string]string{"id": taskID}, nil
}

func (srv *Service) handleTaskGet(params json.RawMessage) (interface{}, *JSONRPCError) {
	// Parse request parameters
	var reqParams struct {
		ID string `json:"id"`
	}

	if err := json.Unmarshal(params, &reqParams); err != nil {
		return nil, &JSONRPCError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    err.Error(),
		}
	}

	if reqParams.ID == "" {
		return nil, &JSONRPCError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    "task ID is required",
		}
	}

	// TODO: Retrieve actual task from storage
	// For now, return a placeholder
	return map[string]interface{}{
		"id": reqParams.ID,
		"status": map[string]string{
			"state": "working",
		},
	}, nil
}

func (srv *Service) handleTaskSend(params json.RawMessage) (interface{}, *JSONRPCError) {
	// Parse request parameters
	var reqParams TaskSendParams

	if err := json.Unmarshal(params, &reqParams); err != nil {
		return nil, &JSONRPCError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    err.Error(),
		}
	}

	if reqParams.ID == "" {
		return nil, &JSONRPCError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    "task ID is required",
		}
	}

	// TODO: Add message to task history

	// Update task status
	status := TaskStatus{
		State:     TaskStateWorking,
		Timestamp: time.Now().Format(time.RFC3339),
	}

	// Send SSE update
	srv.SendTaskUpdate(reqParams.ID, map[string]interface{}{
		"status": status,
	})

	return map[string]string{"status": "accepted"}, nil
}

func (srv *Service) handleTaskSetPushNotification(params json.RawMessage) (any, *JSONRPCError) {
	// Parse request parameters
	var reqParams TaskPushNotificationConfig

	if err := json.Unmarshal(params, &reqParams); err != nil {
		return nil, &JSONRPCError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    err.Error(),
		}
	}

	if reqParams.ID == "" || reqParams.PushNotification.URL == "" {
		return nil, &JSONRPCError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    "task ID and notification URL are required",
		}
	}

	// Register push notification config
	srv.notificationMgr.RegisterPushConfig(reqParams.ID, reqParams.PushNotification)

	return map[string]string{"status": "configured"}, nil
}

func (srv *Service) handleTaskArtifactCreate(params json.RawMessage) (interface{}, *JSONRPCError) {
	// Parse request parameters
	var reqParams struct {
		TaskID   string   `json:"taskId"`
		Artifact Artifact `json:"artifact"`
	}

	if err := json.Unmarshal(params, &reqParams); err != nil {
		return nil, &JSONRPCError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    err.Error(),
		}
	}

	if reqParams.TaskID == "" {
		return nil, &JSONRPCError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    "task ID is required",
		}
	}

	// Generate artifact ID
	artifactID := fmt.Sprintf("artifact-%d", time.Now().UnixNano())

	// TODO: Store the artifact

	// Send SSE update and push notification
	srv.SendTaskUpdate(reqParams.TaskID, map[string]interface{}{
		"artifact": reqParams.Artifact,
	})

	// Send push notification if configured
	if srv.card.Capabilities.PushNotifications {
		srv.notificationMgr.SendTaskArtifactUpdate(
			reqParams.TaskID,
			reqParams.Artifact,
			map[string]any{"artifactId": artifactID},
		)
	}

	return map[string]string{"id": artifactID}, nil
}

func (srv *Service) handleTaskArtifactGet(params json.RawMessage) (interface{}, *JSONRPCError) {
	// Parse request parameters
	var reqParams struct {
		TaskID     string `json:"taskId"`
		ArtifactID string `json:"artifactId"`
	}

	if err := json.Unmarshal(params, &reqParams); err != nil {
		return nil, &JSONRPCError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    err.Error(),
		}
	}

	if reqParams.TaskID == "" || reqParams.ArtifactID == "" {
		return nil, &JSONRPCError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    "task ID and artifact ID are required",
		}
	}

	// TODO: Retrieve the actual artifact
	// For now, return a placeholder
	return map[string]interface{}{
		"id":   reqParams.ArtifactID,
		"name": "example.txt",
	}, nil
}

func (srv *Service) Listen(addr string) error {
	// Register routes before starting the server
	srv.RegisterRoutes()

	return srv.app.Listen(addr, fiber.ListenConfig{
		AutoCertManager: srv.certManager,
	})
}
