package auth

import (
	"crypto/sha256"
	"crypto/subtle"
	"log"
	"os"

	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/keyauth"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type AuthenticationResponse struct {
	Version string               `json:"version"`
	Error   *AuthenticationError `json:"error"`
	ID      *string              `json:"id"`
}

type AuthenticationError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    string `json:"data"`
}

func (e *AuthenticationError) Error() string {
	return e.Message
}

type Authentication struct {
	Schemes     string `json:"schemes,omitempty"`     // Single auth scheme according to A2A spec
	Credentials string `json:"credentials,omitempty"` // Optional credentials
}

// NewKeyAuth creates a new key authentication middleware
func NewKeyAuth() fiber.Handler {
	return keyauth.New(keyauth.Config{
		KeyLookup:  "header:Authorization",
		AuthScheme: "Bearer",
		Validator:  ValidateAPIKey,
		ErrorHandler: func(c fiber.Ctx, err error) error {
			if err == keyauth.ErrMissingOrMalformedAPIKey {
				return c.Status(fiber.StatusUnauthorized).JSON(AuthenticationResponse{
					Version: "2.0",
					Error: &AuthenticationError{
						Code:    -32001,
						Message: "Unauthorized",
						Data:    "Missing or invalid API key",
					},
					ID: nil,
				})
			}
			return c.Status(fiber.StatusUnauthorized).JSON(AuthenticationResponse{
				Version: "2.0",
				Error: &AuthenticationError{
					Code:    -32001,
					Message: "Unauthorized",
					Data:    err.Error(),
				},
				ID: nil,
			})
		},
	})
}

// validateAPIKey checks if the provided API key is valid
func ValidateAPIKey(c fiber.Ctx, key string) (bool, error) {
	// Get API key from environment or config
	apiKey := os.Getenv("API_KEY")
	if apiKey == "" {
		apiKey = tweaker.Value[string]("settings.agent.api_key")
		if apiKey == "" {
			log.Println("Warning: API_KEY not set in environment or configuration")
			return false, keyauth.ErrMissingOrMalformedAPIKey
		}
	}

	// Compare using constant-time comparison to prevent timing attacks
	hashedAPIKey := sha256.Sum256([]byte(apiKey))
	hashedKey := sha256.Sum256([]byte(key))

	if subtle.ConstantTimeCompare(hashedAPIKey[:], hashedKey[:]) == 1 {
		return true, nil
	}
	return false, keyauth.ErrMissingOrMalformedAPIKey
}

// AuthFilter determines which routes should be protected by auth
func AuthFilter(c fiber.Ctx) bool {
	// Public endpoints that don't require authentication
	path := c.Path()

	// Allow agent card and root endpoint without authentication
	if path == "/" || path == "/.well-known/ai-agent.json" {
		return true
	}

	return false
}
