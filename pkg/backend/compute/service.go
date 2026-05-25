package compute

import (
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/backend/compute/schema"
)

/*
Service routes compute schema requests to the appropriate sub-service.
*/
type Service struct {
	handlers map[string]handler
}

type handler interface {
	Request(fiber.Ctx) error
}

/*
NewService creates a new Service with operation, optimizer, and block sub-services.
*/
func NewService() *Service {
	return &Service{
		handlers: map[string]handler{
			"operation": schema.NewOperation(),
			"optimizer": schema.NewOptimizer(),
			"block":     schema.NewBlock(),
		},
	}
}

/*
Request dispatches to the correct sub-service based on the :kind route param.
*/
func (service *Service) Request(ctx fiber.Ctx) error {
	h, ok := service.handlers[ctx.Params("kind")]

	if !ok {
		return ctx.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "unknown kind: " + ctx.Params("kind"),
		})
	}

	return h.Request(ctx)
}
