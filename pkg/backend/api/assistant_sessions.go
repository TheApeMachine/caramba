package api

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/gofiber/fiber/v3"
	_ "github.com/lib/pq"
)

/*
AssistantSessionService handles CRUD for assistant_sessions, assistant_messages,
and the assistant_session_personas join table. Sessions are either team-scoped
(visible to an org) or personal-scoped (visible to one user). Global scope is
intentionally not supported for sessions — conversations are not shared globally.
*/
type AssistantSessionService struct {
	pool *sqlPool
}

type sessionUpsertRequest struct {
	ID         string   `json:"id"`
	Scope      string   `json:"scope"`
	Title      string   `json:"title"`
	WindowSize int      `json:"window_size"`
	PersonaIDs []string `json:"persona_ids"`
}

type messageInsertRequest struct {
	ID          string          `json:"id"`
	SessionID   string          `json:"session_id"`
	Role        string          `json:"role"`
	Parts       json.RawMessage `json:"parts"`
	PersonaID   string          `json:"persona_id"`
	PersonaName string          `json:"persona_name"`
}

func NewAssistantSessionService(databaseURL string) *AssistantSessionService {
	return &AssistantSessionService{pool: newSQLPool(databaseURL)}
}

func (service *AssistantSessionService) CreateSession(ctx fiber.Ctx) error {
	return mutate(ctx, "session", service.insertSession)
}

func (service *AssistantSessionService) UpdateSession(ctx fiber.Ctx) error {
	return mutate(ctx, "session", service.updateSession)
}

func (service *AssistantSessionService) DeleteSession(ctx fiber.Ctx) error {
	return mutate(ctx, "session", service.deleteSession)
}

func (service *AssistantSessionService) CreateMessage(ctx fiber.Ctx) error {
	return mutate(ctx, "message", service.insertMessage)
}

func resolveSessionScope(scope string, identity clerkIdentity) (sql.NullString, sql.NullString, error) {
	switch strings.TrimSpace(scope) {
	case "team":
		trimmed := strings.TrimSpace(identity.orgSlug)

		if trimmed == "" {
			return sql.NullString{}, sql.NullString{}, forbidden(fmt.Errorf("team sessions require an active organization"))
		}

		return sql.NullString{}, sql.NullString{String: trimmed, Valid: true}, nil
	case "personal":
		return sql.NullString{String: identity.subject, Valid: true}, sql.NullString{}, nil
	}

	return sql.NullString{}, sql.NullString{}, forbidden(fmt.Errorf("unknown session scope %q", scope))
}

func (service *AssistantSessionService) insertSession(
	ctx fiber.Ctx, identity clerkIdentity, request sessionUpsertRequest,
) (int64, error) {
	owner, org, err := resolveSessionScope(request.Scope, identity)

	if err != nil {
		return 0, err
	}

	database, err := service.pool.open()

	if err != nil {
		return 0, err
	}

	title := strings.TrimSpace(request.Title)

	if title == "" {
		title = "New conversation"
	}

	now := time.Now().UTC()

	return runWithTxid(ctx, database, func(transaction *sql.Tx) error {
		if _, err := transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO assistant_sessions (
                id, scope, owner_id, organization_slug, title, window_size,
                created_at, updated_at
              ) VALUES ($1,$2,$3,$4,$5,$6,$7,$7)`,
			request.ID, request.Scope, owner, org, title, request.WindowSize, now,
		); err != nil {
			return err
		}

		return insertSessionPersonas(ctx, transaction, request.ID, request.PersonaIDs)
	})
}

func (service *AssistantSessionService) updateSession(
	ctx fiber.Ctx, identity clerkIdentity, request sessionUpsertRequest,
) (int64, error) {
	database, err := service.pool.open()

	if err != nil {
		return 0, err
	}

	return runWithTxid(ctx, database, func(transaction *sql.Tx) error {
		result, err := transaction.ExecContext(
			ctx.Context(),
			`UPDATE assistant_sessions
                SET title=$2, window_size=$3, updated_at=NOW()
              WHERE id=$1
                AND (
                     (scope='personal' AND owner_id=$4)
                  OR (scope='team' AND organization_slug=$5)
                )`,
			request.ID, request.Title, request.WindowSize,
			identity.subject, strings.TrimSpace(identity.orgSlug),
		)

		if err != nil {
			return err
		}

		affected, _ := result.RowsAffected()

		if affected == 0 {
			return forbidden(fmt.Errorf("session not found or not authorized"))
		}

		if _, err := transaction.ExecContext(
			ctx.Context(),
			`DELETE FROM assistant_session_personas WHERE session_id=$1`,
			request.ID,
		); err != nil {
			return err
		}

		return insertSessionPersonas(ctx, transaction, request.ID, request.PersonaIDs)
	})
}

func insertSessionPersonas(
	ctx fiber.Ctx, transaction *sql.Tx, sessionID string, personaIDs []string,
) error {
	for position, personaID := range personaIDs {
		if _, err := transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO assistant_session_personas (session_id, persona_id, position)
              VALUES ($1,$2,$3)
              ON CONFLICT DO NOTHING`,
			sessionID, personaID, position,
		); err != nil {
			return err
		}
	}

	return nil
}

func (service *AssistantSessionService) deleteSession(
	ctx fiber.Ctx, identity clerkIdentity, request sessionUpsertRequest,
) (int64, error) {
	database, err := service.pool.open()

	if err != nil {
		return 0, err
	}

	return runWithTxid(ctx, database, func(transaction *sql.Tx) error {
		_, err := transaction.ExecContext(
			ctx.Context(),
			`DELETE FROM assistant_sessions
              WHERE id=$1
                AND (
                     (scope='personal' AND owner_id=$2)
                  OR (scope='team' AND organization_slug=$3)
                )`,
			request.ID, identity.subject, strings.TrimSpace(identity.orgSlug),
		)

		return err
	})
}

func (service *AssistantSessionService) insertMessage(
	ctx fiber.Ctx, identity clerkIdentity, request messageInsertRequest,
) (int64, error) {
	database, err := service.pool.open()

	if err != nil {
		return 0, err
	}

	parts := request.Parts

	if len(parts) == 0 {
		parts = json.RawMessage("[]")
	}

	return runWithTxid(ctx, database, func(transaction *sql.Tx) error {
		var probe int

		err := transaction.QueryRowContext(
			ctx.Context(),
			`SELECT 1 FROM assistant_sessions
              WHERE id=$1
                AND (
                     (scope='personal' AND owner_id=$2)
                  OR (scope='team' AND organization_slug=$3)
                )`,
			request.SessionID, identity.subject, strings.TrimSpace(identity.orgSlug),
		).Scan(&probe)

		if err == sql.ErrNoRows {
			return forbidden(fmt.Errorf("session not found or not authorized"))
		}

		if err != nil {
			return fmt.Errorf("message session lookup: %w", err)
		}

		_, err = transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO assistant_messages (id, session_id, role, parts, persona_id, persona_name)
              VALUES ($1,$2,$3,$4::jsonb,$5,$6)`,
			request.ID, request.SessionID, request.Role, string(parts),
			nullString(strings.TrimSpace(request.PersonaID)),
			nullString(strings.TrimSpace(request.PersonaName)),
		)

		return err
	})
}
