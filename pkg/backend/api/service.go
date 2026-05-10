package api

import "github.com/gofiber/fiber/v3"

type Service interface {
	Request(ctx fiber.Ctx) error
}
