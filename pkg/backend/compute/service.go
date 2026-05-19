package compute

import (
	"github.com/gofiber/fiber/v3"
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
			"operation": nil,
			"optimizer": nil,
			"block":     nil,
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
