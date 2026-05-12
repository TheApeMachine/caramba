package api

import (
	"net/http/httptest"
	"testing"

	"github.com/gofiber/fiber/v3"
	. "github.com/smartystreets/goconvey/convey"
)

func TestRequireClerkAdmin(t *testing.T) {
	Convey("Given a backend write route protected by Clerk admin", t, func() {
		Convey("It should reject requests without admin privileges", func() {
			app := fiber.New()
			app.Post("/backend/write", RequireClerkAdmin(), func(ctx fiber.Ctx) error {
				return ctx.SendStatus(fiber.StatusNoContent)
			})

			request := httptest.NewRequest("POST", "/backend/write", nil)
			response, err := app.Test(request)

			So(err, ShouldBeNil)
			So(response.StatusCode, ShouldEqual, fiber.StatusForbidden)
		})

		Convey("It should allow requests with admin privileges", func() {
			app := fiber.New()
			app.Use(func(ctx fiber.Ctx) error {
				ctx.Locals("clerkAdmin", true)

				return ctx.Next()
			})
			app.Post("/backend/write", RequireClerkAdmin(), func(ctx fiber.Ctx) error {
				return ctx.SendStatus(fiber.StatusNoContent)
			})

			request := httptest.NewRequest("POST", "/backend/write", nil)
			response, err := app.Test(request)

			So(err, ShouldBeNil)
			So(response.StatusCode, ShouldEqual, fiber.StatusNoContent)
		})
	})
}

func BenchmarkRequireClerkAdmin(b *testing.B) {
	app := fiber.New()
	app.Use(func(ctx fiber.Ctx) error {
		ctx.Locals("clerkAdmin", true)

		return ctx.Next()
	})
	app.Post("/backend/write", RequireClerkAdmin(), func(ctx fiber.Ctx) error {
		return ctx.SendStatus(fiber.StatusNoContent)
	})

	for b.Loop() {
		request := httptest.NewRequest("POST", "/backend/write", nil)

		if response, err := app.Test(request); err != nil || response.StatusCode != fiber.StatusNoContent {
			b.Fatal(err)
		}
	}
}
