package api

import (
	"database/sql"
	"fmt"
	"strings"
	"time"

	"github.com/gofiber/fiber/v3"
	_ "github.com/lib/pq"
)

/*
AssistantPersonaService handles CRUD for assistant_personas, applying scope rules:
global personas require admin; team personas require an active organization;
personal personas are bound to the authenticated user subject.
*/
type AssistantPersonaService struct {
	pool *sqlPool
}

type personaUpsertRequest struct {
	ID           string  `json:"id"`
	Scope        string  `json:"scope"`
	Name         string  `json:"name"`
	SystemPrompt string  `json:"system_prompt"`
	Model        string  `json:"model"`
	Temperature  float64 `json:"temperature"`
	MaxTokens    int     `json:"max_tokens"`
	AdapterType  string  `json:"adapter_type"`
	EndpointURL  string  `json:"endpoint_url"`
}

func NewAssistantPersonaService(databaseURL string) *AssistantPersonaService {
	return &AssistantPersonaService{pool: newSQLPool(databaseURL)}
}

func (service *AssistantPersonaService) Create(ctx fiber.Ctx) error {
	return mutate(ctx, "persona", service.insert)
}

func (service *AssistantPersonaService) Update(ctx fiber.Ctx) error {
	return mutate(ctx, "persona", service.update)
}

func (service *AssistantPersonaService) Delete(ctx fiber.Ctx) error {
	return mutate(ctx, "persona", service.delete)
}

func resolvePersonaScope(scope string, identity clerkIdentity) (sql.NullString, sql.NullString, error) {
	switch strings.TrimSpace(scope) {
	case "global":
		if !identity.isAdmin {
			return sql.NullString{}, sql.NullString{}, forbidden(fmt.Errorf("global personas require admin role"))
		}

		return sql.NullString{}, sql.NullString{}, nil
	case "team":
		trimmed := strings.TrimSpace(identity.orgSlug)

		if trimmed == "" {
			return sql.NullString{}, sql.NullString{}, forbidden(fmt.Errorf("team personas require an active organization"))
		}

		return sql.NullString{}, sql.NullString{String: trimmed, Valid: true}, nil
	case "personal":
		return sql.NullString{String: identity.subject, Valid: true}, sql.NullString{}, nil
	}

	return sql.NullString{}, sql.NullString{}, forbidden(fmt.Errorf("unknown persona scope %q", scope))
}

func (service *AssistantPersonaService) insert(
	ctx fiber.Ctx, identity clerkIdentity, request personaUpsertRequest,
) (int64, error) {
	owner, org, err := resolvePersonaScope(request.Scope, identity)

	if err != nil {
		return 0, err
	}

	name := strings.TrimSpace(request.Name)

	if name == "" {
		return 0, fmt.Errorf("persona name is required")
	}

	database, err := service.pool.open()

	if err != nil {
		return 0, err
	}

	now := time.Now().UTC()

	return runWithTxid(ctx, database, func(transaction *sql.Tx) error {
		_, err := transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO assistant_personas (
                id, scope, owner_id, organization_slug, name, system_prompt,
                model, temperature, max_tokens, adapter_type, endpoint_url,
                created_at, updated_at
              ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$12)`,
			request.ID, request.Scope, owner, org, name,
			request.SystemPrompt, request.Model, request.Temperature,
			request.MaxTokens, request.AdapterType,
			nullString(strings.TrimSpace(request.EndpointURL)), now,
		)

		return err
	})
}

func (service *AssistantPersonaService) update(
	ctx fiber.Ctx, identity clerkIdentity, request personaUpsertRequest,
) (int64, error) {
	owner, org, err := resolvePersonaScope(request.Scope, identity)

	if err != nil {
		return 0, err
	}

	name := strings.TrimSpace(request.Name)

	if name == "" {
		return 0, fmt.Errorf("persona name is required")
	}

	database, err := service.pool.open()

	if err != nil {
		return 0, err
	}

	return runWithTxid(ctx, database, func(transaction *sql.Tx) error {
		_, err := transaction.ExecContext(
			ctx.Context(),
			`UPDATE assistant_personas
                SET name=$2, system_prompt=$3, model=$4, temperature=$5,
                    max_tokens=$6, adapter_type=$7, endpoint_url=$8,
                    updated_at=NOW()
              WHERE id=$1 AND scope=$9
                AND (owner_id IS NOT DISTINCT FROM $10)
                AND (organization_slug IS NOT DISTINCT FROM $11)`,
			request.ID, name, request.SystemPrompt, request.Model,
			request.Temperature, request.MaxTokens, request.AdapterType,
			nullString(strings.TrimSpace(request.EndpointURL)),
			request.Scope, owner, org,
		)

		return err
	})
}

func (service *AssistantPersonaService) delete(
	ctx fiber.Ctx, identity clerkIdentity, request personaUpsertRequest,
) (int64, error) {
	database, err := service.pool.open()

	if err != nil {
		return 0, err
	}

	return runWithTxid(ctx, database, func(transaction *sql.Tx) error {
		_, err := transaction.ExecContext(
			ctx.Context(),
			`DELETE FROM assistant_personas
              WHERE id=$1 AND (
                    (scope='personal' AND owner_id=$2)
                 OR (scope='team'     AND organization_slug=$3)
                 OR (scope='global'   AND $4)
              )`,
			request.ID, identity.subject,
			strings.TrimSpace(identity.orgSlug), identity.isAdmin,
		)

		return err
	})
}
