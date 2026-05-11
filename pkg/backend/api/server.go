package api

import (
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/cors"
	"github.com/gofiber/fiber/v3/middleware/favicon"
	"github.com/gofiber/fiber/v3/middleware/logger"
	recoverer "github.com/gofiber/fiber/v3/middleware/recover"
	"github.com/gofiber/fiber/v3/middleware/timeout"
	"github.com/theapemachine/caramba/pkg/backend/architecture"
	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/backend/modelscope"
)

/*
Server is the main server for the API.
*/
type Server struct {
	app          *fiber.App
	compute      *compute.Service
	architecture *architecture.Service
	modelscope   *modelscope.Service
}

/*
NewServer creates a new Server.
*/
func NewServer() *Server {
	return &Server{
		app: fiber.New(fiber.Config{
			CaseSensitive: true,
			StrictRouting: true,
			ServerHeader:  "Fiber",
			AppName:       "Caramba v1.0.0",
		}),
		compute:      compute.NewService(),
		architecture: architecture.NewService(),
		modelscope:   modelscope.NewService(),
	}
}

/*
Up starts the server and listens for requests.
*/
func (server *Server) Up() error {
	server.app.Use(
		recoverer.New(),
		logger.New(),
		cors.New(),
		favicon.New(),
	)

	wrap := func(h func(fiber.Ctx) error) fiber.Handler {
		return timeout.New(h, timeout.Config{Timeout: 2 * time.Second})
	}

	wrapSlow := func(h func(fiber.Ctx) error) fiber.Handler {
		return timeout.New(h, timeout.Config{Timeout: 30 * time.Second})
	}

	server.app.Get("/backend/compute/:kind", wrap(server.compute.Request))

	server.app.Get("/backend/modelscope", wrap(server.modelscope.List))
	server.app.Get("/backend/modelscope/inspect", wrapSlow(server.modelscope.Inspect))

	server.app.Get("/backend/architecture", wrap(server.architecture.List))
	server.app.Get("/backend/architecture/:name", wrap(server.architecture.Load))
	server.app.Post("/backend/architecture/:name", wrap(server.architecture.Save))

	return server.app.Listen(":8118")
}
