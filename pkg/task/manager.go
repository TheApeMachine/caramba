package task

import (
	"bufio"
	"encoding/json"
	"errors"
	"strings"

	"github.com/gofiber/fiber/v3"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type Manager struct {
	taskStore   TaskStore
	llmProvider provider.ProviderType
}

type ManagerOption func(*Manager)

func NewManager(opts ...ManagerOption) *Manager {
	errnie.Trace("task.NewManager")

	manager := &Manager{}

	for _, opt := range opts {
		opt(manager)
	}

	return manager
}

func (manager *Manager) HandleTask(ctx fiber.Ctx, request *TaskRequest) error {
	switch request.Method {
	case "tasks/send":
		return manager.handleTaskSend(ctx, request)
	}

	return errnie.New(errnie.WithError(errors.New("method not found")))
}

func (manager *Manager) handleTaskSend(ctx fiber.Ctx, request *TaskRequest) error {
	errnie.Trace("task manager.handleTaskSend", "request", request)

	if err := manager.validate(request); err != nil {
		return errnie.New(errnie.WithError(err))
	}

	errnie.Success("request validated")

	messages := make([]provider.Message, 0, len(request.Params.History))

	parts := make([]string, 0)

	for _, message := range request.Params.History {
		for _, part := range message.Parts {
			parts = append(parts, part.Text)
		}

		messages = append(messages, provider.Message{
			Role:    message.Role.String(),
			Content: strings.Join(parts, ""),
		})

	}

	events, err := manager.llmProvider.Generate(provider.ProviderParams{
		Model:       tweaker.GetModel(tweaker.GetProvider()),
		Temperature: tweaker.GetTemperature(),
		TopP:        tweaker.GetTopP(),
		Messages:    messages,
		Stream:      true,
	})

	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	return ctx.SendStreamWriter(func(w *bufio.Writer) {
		for event := range events {
			request.Params.History = append(request.Params.History, Message{
				Role:  MessageRole(event.Message.Role),
				Parts: []Part{{Text: event.Message.Content}},
			})

			response := NewTaskResponse(request.Params)
			buf, err := json.Marshal(response)

			if err != nil {
				errnie.New(errnie.WithError(err))
				return
			}

			if _, err := w.Write(buf); err != nil {
				errnie.New(errnie.WithError(err))
			}

			if err := w.Flush(); err != nil {
				errnie.New(errnie.WithError(err))
			}
		}
	})
}

func (manager *Manager) validate(request *TaskRequest) error {
	if manager == nil {
		return errnie.New(errnie.WithError(errors.New("manager not set")))
	}

	if manager.llmProvider == nil {
		return errnie.New(errnie.WithError(errors.New("llm provider not set")))
	}

	if manager.taskStore == nil {
		return errnie.New(errnie.WithError(errors.New("task store not set")))
	}

	if request == nil {
		return errnie.New(errnie.WithError(errors.New("request not set")))
	}

	if request.Params.History == nil {
		return errnie.New(errnie.WithError(errors.New("history not set")))
	}

	return nil
}

func WithTaskStore(taskStore TaskStore) ManagerOption {
	return func(manager *Manager) {
		manager.taskStore = taskStore
	}
}

func WithLLMProvider(llmProvider provider.ProviderType) ManagerOption {
	return func(manager *Manager) {
		manager.llmProvider = llmProvider
	}
}
