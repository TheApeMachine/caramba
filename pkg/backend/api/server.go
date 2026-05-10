package api

import (
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/cors"
	"github.com/gofiber/fiber/v3/middleware/favicon"
	"github.com/gofiber/fiber/v3/middleware/logger"
	recoverer "github.com/gofiber/fiber/v3/middleware/recover"
	"github.com/gofiber/fiber/v3/middleware/timeout"
	"github.com/theapemachine/caramba/backend/compute"
)

/*
Server is the main server for the API.
*/
type Server struct {
	app      *fiber.App
	handlers map[string]Service
}

/*
NewServer creates a new Server with the default handlers.
*/
func NewServer() *Server {
	return &Server{
		app: fiber.New(fiber.Config{
			CaseSensitive: true,
			StrictRouting: true,
			ServerHeader:  "Fiber",
			AppName:       "Test App v1.0.1",
		}),
		handlers: map[string]Service{
			"operation": compute.NewService(),
		},
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

	server.app.Get(
		"/backend/compute/:operation",
		timeout.New(func(ctx fiber.Ctx) (err error) {
			handler := server.handlers[ctx.Params("operation")]

			if handler == nil {
				return ctx.Status(fiber.StatusNotFound).JSON(fiber.Map{
					"error": "Operation not found",
				})
			}

			if err = handler.Request(ctx); err != nil {
				return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
					"error": err.Error(),
				})
			}

			return nil
		}, timeout.Config{
			Timeout: 2 * time.Second,
		}))

	return server.app.Listen(":8118")
}
