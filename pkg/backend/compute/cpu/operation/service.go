package operation

import (
	"os"

	"github.com/gofiber/fiber/v3"
)

type OperationSchema struct {
	Name         string   `json:"name"`
	Label        string   `json:"label"`
	Description  string   `json:"description"`
	InitialWidth int      `json:"initial_width"`
	Inputs       []string `json:"inputs"`
	Outputs      []string `json:"outputs"`
}

type Service struct {
	operations map[string]Operation
}

func NewService() *Service {
	return &Service{
		operations: make(map[string]Operation),
	}
}

/*
Request walks the operation directories to find all operations
dynamically and hands them back to the caller. This is used to
populate the frontend's node graph editor, where users can
visually compose architectures.
*/
func (service *Service) Request(ctx fiber.Ctx) error {
	operations := make(map[string]OperationSchema)

	dir, err := os.Open(".")

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	defer dir.Close()

	files, err := dir.Readdir(0)

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		operation := service.operations[file.Name()]

		if operation == nil {
			continue
		}

		operations[file.Name()] = OperationSchema{
			Name: file.Name(),
		}
		_ = operation
	}

	return ctx.JSON(operations)
}
