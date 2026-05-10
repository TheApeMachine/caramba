package compute

import (
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/backend/compute/cpu/operation"
)

/*
Service provides handlers for the compute API.
*/
type Service struct {
	operations map[string]operation.Operation
}

/*
NewService creates a new Service with the default operations.
*/
func NewService() *Service {
	return &Service{}
}

/*
Handle computes the result of the operation.
*/
func (service *Service) Handle(ctx fiber.Ctx) error {
	return ctx.SendString("Hello, World!")
}

/*
Request walks the operation directories to find all operations
dynamically and hands them back to the caller. This is used to
populate the frontend's node graph editor, where users can
visually compose architectures.
*/
func (service *Service) Request(ctx fiber.Ctx) error {
	switch ctx.Params("operation") {
	case "operation":
		return operation.NewService().Request(ctx)
	default:
		return ctx.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "Operation not found",
		})
	}
}
