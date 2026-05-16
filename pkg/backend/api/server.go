package api

import (
	"context"
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
	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/caramba/pkg/devteam"
	"github.com/theapemachine/caramba/pkg/qpool"
)

/*
Server is the main server for the API.
*/
type Server struct {
	app               *fiber.App
	compute           *compute.Service
	architecture      *architecture.Service
	modelscope        *modelscope.Service
	researchProjects  *ResearchProjectService
	assistantPersonas *AssistantPersonaService
	assistantSessions *AssistantSessionService
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
		compute:           compute.NewService(),
		architecture:      architecture.NewService(),
		modelscope:        modelscope.NewService(),
		researchProjects:  NewResearchProjectService(config.NewDevTeamConfig().DatabaseURL),
		assistantPersonas: NewAssistantPersonaService(config.NewDevTeamConfig().DatabaseURL),
		assistantSessions: NewAssistantSessionService(config.NewDevTeamConfig().DatabaseURL),
	}
}

/*
Up starts the server, launches the AI dev team orchestrator if configured, and
listens for requests.
*/
func (server *Server) Up() error {
	server.app.Use(
		recoverer.New(),
		logger.New(),
		cors.New(),
		favicon.New(),
	)

	clerkConfig := config.NewClerkConfig()
	server.app.Use("/backend", RequireClerkSession(clerkConfig))

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
	server.app.Post("/backend/architecture/:name", RequireClerkAdmin(), wrap(server.architecture.Save))
	server.app.Post("/backend/research-projects", RequireClerkAdmin(), wrap(server.researchProjects.Create))

	server.app.Post("/backend/assistant/personas", wrap(server.assistantPersonas.Create))
	server.app.Put("/backend/assistant/personas", wrap(server.assistantPersonas.Update))
	server.app.Delete("/backend/assistant/personas", wrap(server.assistantPersonas.Delete))

	server.app.Post("/backend/assistant/sessions", wrap(server.assistantSessions.CreateSession))
	server.app.Put("/backend/assistant/sessions", wrap(server.assistantSessions.UpdateSession))
	server.app.Delete("/backend/assistant/sessions", wrap(server.assistantSessions.DeleteSession))
	server.app.Post("/backend/assistant/messages", wrap(server.assistantSessions.CreateMessage))

	devteamCfg := config.NewDevTeamConfig()

	if devteamCfg.Active {
		orchestrator, err := devteam.NewOrchestrator(context.Background(), devteamCfg)

		if err == nil {
			workerPool := qpool.NewQ(
				context.Background(),
				1,
				1,
				&qpool.Config{
					SchedulingTimeout:  24 * time.Hour,
					JobChannelCapacity: 1,
					Scaler:             nil,
				},
			)
			workerPool.Schedule(
				"backend.devteam.orchestrator",
				func(context.Context) (any, error) {
					return nil, orchestrator.Run()
				},
				qpool.WithExecTimeout(24*time.Hour),
			)
		}
	}

	return server.app.Listen(":8118")
}
