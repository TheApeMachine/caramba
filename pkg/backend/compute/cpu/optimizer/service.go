package optimizer

import (
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/asset"
)

/*
Service serves optimizer schemas to the frontend node graph editor.
*/
type Service struct{}

/*
NewService creates a new Service.
*/
func NewService() *Service {
	return &Service{}
}

/*
Request returns all optimizer schemas as JSON, keyed by op identifier.
*/
func (service *Service) Request(ctx fiber.Ctx) error {
	schemas, err := asset.Walk("template/optimizer")

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return ctx.JSON(schemas)
}
