package manager

import (
	"bufio"
	"encoding/json"
	"errors"

	"github.com/gofiber/fiber/v3"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/task"
)

type Manager struct {
	taskStore   task.TaskStore
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

func (manager *Manager) HandleTask(ctx fiber.Ctx, request *task.TaskRequest) error {
	switch request.Method {
	case "tasks/send":
		return manager.handleTaskSend(ctx, request)
	}

	return errnie.New(errnie.WithError(errors.New("method not found")))
}

func (manager *Manager) handleTaskSend(ctx fiber.Ctx, request *task.TaskRequest) error {
	errnie.Trace("task manager.handleTaskSend", "request", request)

	if err := manager.validate(request); err != nil {
		return errnie.New(errnie.WithError(err))
	}

	chunks, err := manager.llmProvider.Generate(
		ctx, request,
	)

	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	return ctx.SendStreamWriter(func(w *bufio.Writer) {
		for chunk := range chunks {
			buf, err := json.Marshal(chunk)

			if err != nil {
				errnie.New(errnie.WithError(err))
				return
			}

			if _, err := w.Write(buf); err != nil {
				errnie.New(errnie.WithError(err))
			}
		}
	})
}

func (manager *Manager) validate(request *task.TaskRequest) error {
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

func WithTaskStore(taskStore task.TaskStore) ManagerOption {
	return func(manager *Manager) {
		manager.taskStore = taskStore
	}
}

func WithLLMProvider(llmProvider provider.ProviderType) ManagerOption {
	return func(manager *Manager) {
		manager.llmProvider = llmProvider
	}
}
