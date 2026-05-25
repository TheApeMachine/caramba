package schema

import (
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/asset"
)

/*
Optimizer serves optimizer schemas to the frontend node graph editor.
*/
type Optimizer struct{}

/*
NewOptimizer creates an Optimizer schema handler.
*/
func NewOptimizer() *Optimizer {
	return &Optimizer{}
}

/*
Request returns all optimizer schemas as JSON, keyed by op identifier.
*/
func (optimizer *Optimizer) Request(ctx fiber.Ctx) error {
	schemas, err := asset.Walk("template/optimizer")

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return ctx.JSON(schemas)
}
