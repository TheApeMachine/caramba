package modelscope

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/gofiber/fiber/v3"
)

const modelsDir = "models"

var supportedExts = map[string]bool{
	".gguf":         true,
	".safetensors":  true,
	".onnx":         true,
	".pt":           true,
	".pth":          true,
}

/*
Service inspects model files and returns a GraphData payload that the
modelscope frontend renderer consumes directly.
*/
type Service struct{}

/*
NewService creates a new Service, ensuring the models directory exists.
*/
func NewService() *Service {
	_ = os.MkdirAll(modelsDir, 0o755)
	return &Service{}
}

/*
List returns the names of all model files found in the models directory.
GET /backend/modelscope
*/
func (service *Service) List(ctx fiber.Ctx) error {
	entries, err := os.ReadDir(modelsDir)

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	names := make([]string, 0, len(entries))

	for _, entry := range entries {
		if !entry.IsDir() && supportedExts[strings.ToLower(filepath.Ext(entry.Name()))] {
			names = append(names, entry.Name())
		}
	}

	return ctx.JSON(names)
}

/*
Inspect reads the model header at the given path and returns GraphData JSON.
GET /backend/modelscope/inspect?path=<absolute-path>
*/
func (service *Service) Inspect(ctx fiber.Ctx) error {
	path := ctx.Query("path")

	if path == "" {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "missing query param: path",
		})
	}

	f, err := os.Open(path)

	if err != nil {
		return ctx.Status(fiber.StatusUnprocessableEntity).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	defer f.Close()

	graph, err := service.parse(filepath.Base(path), f)

	if err != nil {
		return ctx.Status(fiber.StatusUnprocessableEntity).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return ctx.JSON(graph)
}

func (service *Service) parse(name string, r io.Reader) (GraphData, error) {
	ext := strings.ToLower(filepath.Ext(name))

	switch ext {
	case ".gguf":
		return parseGGUFReader(r)
	case ".safetensors":
		return parseSafeTensorsReader(r)
	default:
		return GraphData{}, fmt.Errorf("unsupported model format: %q", ext)
	}
}

