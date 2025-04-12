package provider

import (
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/task"
)

type ProviderType interface {
	Generate(fiber.Ctx, *task.TaskRequest) (<-chan *task.TaskResponse, error)
	Stream(fiber.Ctx, *task.TaskRequest) (<-chan *task.TaskResponse, error)
}

type EmbedderType interface {
	Embed(fiber.Ctx, *task.TaskRequest) ([]float64, error)
}
