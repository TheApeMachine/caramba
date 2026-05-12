package api

import (
	"strings"

	"github.com/clerk/clerk-sdk-go/v2"
	clerkjwt "github.com/clerk/clerk-sdk-go/v2/jwt"
	"github.com/gofiber/fiber/v3"

	"github.com/theapemachine/caramba/pkg/config"
)

/*
RequireClerkSession validates Authorization: Bearer <Clerk session JWT> on /backend
routes when Clerk is active and secret_key is configured.

Sets Fiber locals: clerkSubject (string JWT sub), clerkAdmin (bool) when subject
matches clerk.admin_subject_ids or session active org matches
clerk.privileged_organization_slug with org admin role (org:admin / admin).
*/
func RequireClerkSession(clerkConfig *config.ClerkConfig) fiber.Handler {
	if !clerkConfig.Active {
		return func(ctx fiber.Ctx) error {
			return ctx.Next()
		}
	}

	if strings.TrimSpace(clerkConfig.SecretKey) == "" {
		return func(ctx fiber.Ctx) error {
			if !clerkConfig.RequireAuth {
				return ctx.Next()
			}

			return ctx.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{
				"error": "clerk secret key is required",
			})
		}
	}

	clerk.SetKey(strings.TrimSpace(clerkConfig.SecretKey))

	return func(ctx fiber.Ctx) error {
		token := bearerToken(strings.TrimSpace(ctx.Get("Authorization")))

		if token == "" {
			return ctx.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "missing bearer token",
			})
		}

		claims, verifyErr := clerkjwt.Verify(ctx.Context(), &clerkjwt.VerifyParams{
			Token: token,
		})

		if verifyErr != nil {
			return ctx.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "invalid session token",
			})
		}

		ctx.Locals("clerkSubject", claims.Subject)
		ctx.Locals("clerkOrganizationSlug", claims.ActiveOrganizationSlug)
		ctx.Locals("clerkOrganizationRole", claims.ActiveOrganizationRole)
		ctx.Locals(
			"clerkAdmin",
			clerkConfig.SubjectHasElevatedAdminPrivileges(
				claims.Subject,
				claims.ActiveOrganizationSlug,
				claims.ActiveOrganizationRole,
			),
		)

		return ctx.Next()
	}
}

func RequireClerkAdmin() fiber.Handler {
	return func(ctx fiber.Ctx) error {
		clerkAdmin, ok := ctx.Locals("clerkAdmin").(bool)

		if ok && clerkAdmin {
			return ctx.Next()
		}

		return ctx.Status(fiber.StatusForbidden).JSON(fiber.Map{
			"error": "admin privileges required",
		})
	}
}

func bearerToken(authorizationHeader string) string {
	const bearerPrefix = "Bearer "

	if !strings.HasPrefix(authorizationHeader, bearerPrefix) {
		return ""
	}

	return strings.TrimSpace(strings.TrimPrefix(authorizationHeader, bearerPrefix))
}
