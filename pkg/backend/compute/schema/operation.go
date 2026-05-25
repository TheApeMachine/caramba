package schema

import (
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/asset"
)

/*
Operation serves operation schemas to the frontend node graph editor.
Schemas are derived from embedded YAML manifests in manifesto/asset.
*/
type Operation struct{}

/*
NewOperation creates an Operation schema handler.
*/
func NewOperation() *Operation {
	return &Operation{}
}

/*
Request returns all operation schemas as JSON, keyed by op identifier.
*/
func (operation *Operation) Request(ctx fiber.Ctx) error {
	schemas, err := asset.Walk("template/operation")

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return ctx.JSON(schemas)
}
