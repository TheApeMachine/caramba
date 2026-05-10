package architecture

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/gofiber/fiber/v3"
)

const storageDir = "architectures"

/*
Service handles saving and loading named architecture graphs.
An architecture is a serialized Flume NodeMap — the complete
graph state exported from the frontend node editor.
*/
type Service struct{}

/*
NewService creates a new Service, ensuring the storage directory exists.
*/
func NewService() *Service {
	_ = os.MkdirAll(storageDir, 0o755)
	return &Service{}
}

/*
List returns the names of all saved architectures.
*/
func (service *Service) List(ctx fiber.Ctx) error {
	entries, err := os.ReadDir(storageDir)

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	names := make([]string, 0, len(entries))

	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".json") {
			names = append(names, strings.TrimSuffix(entry.Name(), ".json"))
		}
	}

	return ctx.JSON(names)
}

/*
Load returns the NodeMap JSON for a named architecture.
*/
func (service *Service) Load(ctx fiber.Ctx) error {
	name := ctx.Params("name")
	data, err := os.ReadFile(filepath.Join(storageDir, name+".json"))

	if err != nil {
		return ctx.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "architecture not found: " + name,
		})
	}

	ctx.Set("Content-Type", "application/json")
	return ctx.Send(data)
}

/*
Save persists a NodeMap JSON under the given name.
*/
func (service *Service) Save(ctx fiber.Ctx) error {
	name := ctx.Params("name")

	var payload json.RawMessage

	if err := ctx.Bind().JSON(&payload); err != nil {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	data, err := json.MarshalIndent(payload, "", "  ")

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	if err := os.WriteFile(filepath.Join(storageDir, name+".json"), data, 0o644); err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return ctx.JSON(fiber.Map{"saved": name})
}
