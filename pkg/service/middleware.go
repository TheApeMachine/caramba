/*
Package service provides middleware components for the application, including
authentication, rate limiting, logging, and error handling.
*/

package service

import (
	"fmt"
	"time"

	"github.com/getsentry/sentry-go"
	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/compress"
	"github.com/gofiber/fiber/v3/middleware/keyauth"
	"github.com/gofiber/fiber/v3/middleware/limiter"
	"github.com/gofiber/fiber/v3/middleware/logger"
	"github.com/gofiber/fiber/v3/middleware/recover"
	"github.com/gofiber/fiber/v3/middleware/requestid"
	"github.com/theapemachine/caramba/pkg/auth"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/service/types"
)

/*
Middleware manages the application's HTTP middleware stack, providing a centralized
way to configure and apply middleware components like authentication, logging,
and error handling.

Example:

	app := fiber.New()
	middleware := NewMiddleware(app)
	middleware.Register()
*/
type Middleware struct {
	app *fiber.App
}

/*
NewMiddleware creates a new middleware manager for the given Fiber application.
*/
func NewMiddleware() *Middleware {
	return &Middleware{}
}

/*
Register applies all middleware components to the Fiber application in the correct
order. This includes request ID generation, logging, error recovery, compression,
rate limiting, and authentication.
*/
func (middleware *Middleware) Register(app *fiber.App) {
	middleware.app = app
	middleware.app.Use(requestid.New())

	middleware.app.Use(logger.New(logger.Config{
		Format:     "[${time}] ${status} - ${method} ${path} (${ip}) ${latency}\n",
		TimeFormat: "2006/01/02 15:04:05",
		TimeZone:   "Local",
	}))

	middleware.app.Use(recover.New(recover.Config{
		EnableStackTrace: true,
		StackTraceHandler: func(c fiber.Ctx, e any) {
			// Log the error with our error system
			err := errnie.New(
				errnie.WithMessage(fmt.Sprintf("Panic recovered: %v", e)),
				errnie.WithType(errnie.SystemError),
				errnie.WithLevel(sentry.LevelError),
			)

			errnie.New(errnie.WithError(err))
		},
	}))

	middleware.app.Use(compress.New(compress.Config{
		Level: compress.LevelBestSpeed,
	}))

	middleware.app.Use(limiter.New(limiter.Config{
		Max:        100,             // Max number of requests
		Expiration: 1 * time.Minute, // Per minute
		KeyGenerator: func(c fiber.Ctx) string {
			// Use IP as the key for rate limiting
			return c.IP()
		},
		LimitReached: func(c fiber.Ctx) error {
			err := errnie.New(
				errnie.WithMessage("Rate limit exceeded"),
				errnie.WithType(errnie.RateLimitError),
				errnie.WithStatus(errnie.TooManyRequestsStatus),
			)

			return c.Status(err.Status()).JSON(types.JSONRPCResponse{
				Version: "2.0",
				Error: &types.JSONRPCError{
					Code:    -32429,
					Message: "Too Many Requests",
					Data:    "Rate limit exceeded. Please try again later.",
				},
			})
		},
	}))

	// Apply key authentication middleware with filter for public endpoints
	middleware.app.Use(keyauth.New(keyauth.Config{
		KeyLookup:  "header:Authorization",
		AuthScheme: "Bearer",
		Validator:  auth.ValidateAPIKey,
		Next:       auth.AuthFilter,
		ErrorHandler: func(c fiber.Ctx, err error) error {
			if err == keyauth.ErrMissingOrMalformedAPIKey {
				authErr := errnie.New(
					errnie.WithMessage("Unauthorized: missing or invalid API key"),
					errnie.WithType(errnie.AuthenticationError),
					errnie.WithStatus(errnie.UnauthorizedStatus),
				)

				return c.Status(authErr.Status()).JSON(types.JSONRPCResponse{
					Version: "2.0",
					Error: &types.JSONRPCError{
						Code:    -32001,
						Message: "Unauthorized",
						Data:    "Missing or invalid API key",
					},
				})
			}

			authErr := errnie.New(
				errnie.WithMessage(fmt.Sprintf("Unauthorized: %s", err.Error())),
				errnie.WithType(errnie.AuthenticationError),
				errnie.WithStatus(errnie.UnauthorizedStatus),
				errnie.WithError(err),
			)

			return c.Status(authErr.Status()).JSON(types.JSONRPCResponse{
				Version: "2.0",
				Error: &types.JSONRPCError{
					Code:    -32001,
					Message: "Unauthorized",
					Data:    err.Error(),
				},
			})
		},
	}))
}
