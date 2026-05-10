package operation

import (
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/asset"
)

/*
Service serves operation schemas to the frontend node graph editor.
Schemas are derived dynamically from the embedded YAML manifests.
*/
type Service struct{}

/*
NewService creates a new Service.
*/
func NewService() *Service {
	return &Service{}
}

/*
Request returns all operation schemas as JSON, keyed by op identifier.
*/
func (service *Service) Request(ctx fiber.Ctx) error {
	schemas, err := asset.Walk("template/operation")

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return ctx.JSON(schemas)
}
