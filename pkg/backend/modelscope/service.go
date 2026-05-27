package modelscope

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/gofiber/fiber/v3"
)

const defaultModelsDir = "models"

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

The directory it scans is resolved at construction time, in this order:
  1. CARAMBA_MODELS_DIR env var (absolute or relative path)
  2. "./models" relative to the process working directory

Either way, the resolved absolute path is logged at startup so it's
clear which directory the API is actually reading from.
*/
type Service struct {
	modelsDir string
}

/*
NewService creates a new Service, resolving the models directory and
ensuring it exists.
*/
func NewService() *Service {
	dir := strings.TrimSpace(os.Getenv("CARAMBA_MODELS_DIR"))

	if dir == "" {
		dir = defaultModelsDir
	}

	resolved, err := filepath.Abs(dir)

	if err != nil {
		log.Printf("modelscope: cannot resolve models dir %q: %v", dir, err)
		resolved = dir
	}

	if err := os.MkdirAll(resolved, 0o755); err != nil {
		log.Printf("modelscope: cannot create models dir %q: %v", resolved, err)
	}

	log.Printf("modelscope: scanning models directory %q", resolved)

	return &Service{modelsDir: resolved}
}

/*
List returns the names of all model files found in the models directory.
GET /backend/modelscope
*/
func (service *Service) List(ctx fiber.Ctx) error {
	entries, err := os.ReadDir(service.modelsDir)

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error":      err.Error(),
			"models_dir": service.modelsDir,
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
The path query parameter is resolved against the configured models
directory so the frontend can keep sending the relative form
("models/foo.safetensors") and not need to know where the binary
actually runs from.
GET /backend/modelscope/inspect?path=<path>
*/
func (service *Service) Inspect(ctx fiber.Ctx) error {
	requested := strings.TrimSpace(ctx.Query("path"))

	if requested == "" {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "missing query param: path",
		})
	}

	resolved := service.resolveInspectPath(requested)

	f, err := os.Open(resolved)

	if err != nil {
		return ctx.Status(fiber.StatusUnprocessableEntity).JSON(fiber.Map{
			"error":         err.Error(),
			"requested":     requested,
			"resolved_path": resolved,
		})
	}

	defer f.Close()

	graph, err := service.parse(filepath.Base(resolved), f)

	if err != nil {
		return ctx.Status(fiber.StatusUnprocessableEntity).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return ctx.JSON(graph)
}

/*
resolveInspectPath maps the requested path to an absolute file:
  - absolute paths pass through
  - paths starting with "models/" are joined against the configured
    models dir using the segment after the prefix
  - everything else is joined against the configured models dir
*/
func (service *Service) resolveInspectPath(requested string) string {
	if filepath.IsAbs(requested) {
		return requested
	}

	trimmed := strings.TrimPrefix(requested, "./")
	trimmed = strings.TrimPrefix(trimmed, "models/")

	return filepath.Join(service.modelsDir, trimmed)
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

