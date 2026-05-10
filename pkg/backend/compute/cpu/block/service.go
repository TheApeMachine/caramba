package block

import (
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/asset"
)

/*
Service serves block schemas to the frontend node graph editor.
Blocks are pre-wired groups of operations that appear as a single
collapsed node in the graph with exposed external ports only.
*/
type Service struct{}

/*
NewService creates a new Service.
*/
func NewService() *Service {
	return &Service{}
}

/*
Request returns all block schemas as JSON, keyed by op identifier.
*/
func (service *Service) Request(ctx fiber.Ctx) error {
	schemas, err := asset.Walk("template/block")

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return ctx.JSON(schemas)
}
