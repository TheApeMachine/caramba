package errnie

import (
	"context"
	"time"

	"github.com/getsentry/sentry-go"
	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/adaptor"
	"github.com/gofiber/utils"
	"github.com/spf13/viper"
)

var v *viper.Viper

func init() {
	v = viper.GetViper()
}

// Config defines the config for middleware.
type Config struct {
	Enabled         bool
	Repanic         bool
	WaitForDelivery bool
	Timeout         time.Duration
	DSN             string
	Environment     string
	Release         string
	SampleRate      float64
}

// Helper function to set default values
func configDefault(config ...Config) Config {
	if len(config) < 1 {
		return Config{
			Enabled:     v.GetBool("settings.sentry.enabled"),
			DSN:         v.GetString("settings.sentry.dsn"),
			Environment: v.GetString("settings.sentry.environment"),
			Release:     v.GetString("settings.sentry.release"),
			SampleRate:  v.GetFloat64("settings.sentry.sample_rate"),
		}
	}

	cfg := config[0]

	if cfg.Timeout == 0 {
		cfg.Timeout = time.Second * 2
	}

	return cfg
}

// New creates a new middleware handler
func NewSentryMiddleware(config ...Config) fiber.Handler {
	// Set default config
	cfg := configDefault(config...)

	// Return new handler
	return func(c fiber.Ctx) error {
		// Convert fiber request to http request
		r, err := adaptor.ConvertRequest(c, true)

		if err != nil {
			return err
		}

		// Init sentry hub
		hub := sentry.CurrentHub().Clone()
		scope := hub.Scope()
		scope.SetRequest(r)
		scope.SetRequestBody(utils.SafeBytes(c.Body()))
		c.Locals("sentry.hub", hub)

		// Catch panics
		defer func() {
			if err := recover(); err != nil {
				eventID := hub.RecoverWithContext(
					context.WithValue(context.Background(), sentry.RequestContextKey, c),
					err,
				)

				if eventID != nil && cfg.WaitForDelivery {
					hub.Flush(cfg.Timeout)
				}

				if cfg.Repanic {
					panic(err)
				}
			}
		}()

		// Return err if exist, else move to next handler
		return c.Next()
	}
}

func GetHubFromContext(ctx fiber.Ctx) *sentry.Hub {
	return ctx.Locals("sentry.hub").(*sentry.Hub)
}
