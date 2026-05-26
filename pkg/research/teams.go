package research

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/backend/apidata"
)

/*
TeamService owns CRUD for teams within an organization. Teams sit between a
Clerk organization and a research project; a user can belong to many teams,
and a project belongs to at most one team.
*/
type TeamService struct {
	pool *apidata.SQLPool
}

type createTeamRequest struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Slug        string `json:"slug"`
	Description string `json:"description"`
	Color       string `json:"color"`
	Emoji       string `json:"emoji"`
	PrivacyMode string `json:"privacy_mode"`
}

type updateTeamRequest struct {
	ID          string  `json:"id"`
	Name        *string `json:"name"`
	Description *string `json:"description"`
	Color       *string `json:"color"`
	Emoji       *string `json:"emoji"`
	PrivacyMode *string `json:"privacy_mode"`
}

type teamListRow struct {
	ID               string    `json:"id"`
	OrganizationSlug string    `json:"organization_slug"`
	Name             string    `json:"name"`
	Slug             string    `json:"slug"`
	Description      string    `json:"description"`
	Color            string    `json:"color"`
	Emoji            string    `json:"emoji"`
	PrivacyMode      string    `json:"privacy_mode"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
	Role             string    `json:"role"`
}

func NewTeamService(databaseURL string) *TeamService {
	return &TeamService{pool: apidata.NewSQLPool(databaseURL)}
}

/*
Create inserts a team row and an owner membership for the calling user.
*/
func (service *TeamService) Create(ctx fiber.Ctx) error {
	return apidata.Mutate(ctx, "team", service.create)
}

/*
Update patches a team's mutable fields. Only fields the caller explicitly
sends in the JSON payload are written; missing fields are left alone.
Requires the caller to be a member of the team.
*/
func (service *TeamService) Update(ctx fiber.Ctx) error {
	return apidata.Mutate(ctx, "team update", service.update)
}

/*
List returns every team in the caller's current organization, annotated with the
caller's role in each team (or empty when not a member yet).
*/
func (service *TeamService) List(ctx fiber.Ctx) error {
	identity, err := apidata.ReadClerkIdentity(ctx)

	if err != nil {
		return ctx.Status(fiber.StatusUnauthorized).JSON(fiber.Map{"error": err.Error()})
	}

	database, err := service.pool.Open()

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	orgSlug := organizationSlugFromIdentity(identity)

	rows, err := database.QueryContext(
		ctx.Context(),
		`SELECT t.id, t.organization_slug, t.name, t.slug, t.description,
            t.color, t.emoji, t.privacy_mode,
            t.created_at, t.updated_at,
            COALESCE(m.role, '') AS role
       FROM teams t
       LEFT JOIN team_memberships m
         ON m.team_id = t.id AND m.user_id = $2
      WHERE t.organization_slug = $1
      ORDER BY t.created_at ASC`,
		orgSlug,
		identity.Subject,
	)

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": fmt.Sprintf("team list: %s", err.Error())})
	}

	defer rows.Close()

	teams := []teamListRow{}

	for rows.Next() {
		row := teamListRow{}

		if err := rows.Scan(
			&row.ID,
			&row.OrganizationSlug,
			&row.Name,
			&row.Slug,
			&row.Description,
			&row.Color,
			&row.Emoji,
			&row.PrivacyMode,
			&row.CreatedAt,
			&row.UpdatedAt,
			&row.Role,
		); err != nil {
			return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": fmt.Sprintf("team list scan: %s", err.Error())})
		}

		teams = append(teams, row)
	}

	if err := rows.Err(); err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": fmt.Sprintf("team list rows: %s", err.Error())})
	}

	body, err := json.Marshal(teams)

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": fmt.Sprintf("team list encode: %s", err.Error())})
	}

	ctx.Set("Content-Type", "application/json")
	return ctx.Send(body)
}

func (service *TeamService) create(
	ctx fiber.Ctx,
	identity apidata.ClerkIdentity,
	request createTeamRequest,
) (int64, error) {
	database, err := service.pool.Open()

	if err != nil {
		return 0, err
	}

	teamID := strings.TrimSpace(request.ID)

	if teamID == "" {
		return 0, fmt.Errorf("team id is required")
	}

	name := strings.TrimSpace(request.Name)

	if name == "" {
		return 0, fmt.Errorf("team name is required")
	}

	orgSlug := organizationSlugFromIdentity(identity)

	return apidata.RunWithTxid(ctx, database, func(transaction *sql.Tx) error {
		slug, err := service.resolveTeamSlug(
			transaction,
			ctx,
			orgSlug,
			strings.TrimSpace(request.Slug),
			name,
		)

		if err != nil {
			return err
		}

		now := time.Now().UTC()

		privacyMode := strings.TrimSpace(request.PrivacyMode)

		if privacyMode != "shared" && privacyMode != "local" {
			privacyMode = "shared"
		}

		_, err = transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO teams (
        id, organization_slug, name, slug, description, color, emoji,
        privacy_mode, created_at, updated_at
      )
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)`,
			teamID,
			orgSlug,
			name,
			slug,
			strings.TrimSpace(request.Description),
			strings.TrimSpace(request.Color),
			strings.TrimSpace(request.Emoji),
			privacyMode,
			now,
		)

		if err != nil {
			return fmt.Errorf("team insert: %w", err)
		}

		_, err = transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO team_memberships (team_id, user_id, role, created_at)
       VALUES ($1, $2, 'owner', $3)
       ON CONFLICT (team_id, user_id) DO NOTHING`,
			teamID,
			identity.Subject,
			now,
		)

		if err != nil {
			return fmt.Errorf("team owner membership insert: %w", err)
		}

		return nil
	})
}

func (service *TeamService) resolveTeamSlug(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	organizationSlug string,
	requestedSlug string,
	teamName string,
) (string, error) {
	base := deriveProjectSlug(teamName)

	if requestedSlug != "" {
		base = deriveProjectSlug(requestedSlug)
	}

	candidate := base

	for suffix := 0; suffix < 100; suffix++ {
		if suffix > 0 {
			candidate = fmt.Sprintf("%s-%d", base, suffix+1)
		}

		var exists bool

		err := transaction.QueryRowContext(
			ctx.Context(),
			`SELECT EXISTS (
        SELECT 1 FROM teams
        WHERE organization_slug = $1 AND slug = $2
      )`,
			organizationSlug,
			candidate,
		).Scan(&exists)

		if err != nil {
			return "", fmt.Errorf("team slug lookup: %w", err)
		}

		if !exists {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("could not allocate a unique team slug")
}

func (service *TeamService) update(
	ctx fiber.Ctx,
	identity apidata.ClerkIdentity,
	request updateTeamRequest,
) (int64, error) {
	database, err := service.pool.Open()

	if err != nil {
		return 0, err
	}

	teamID := strings.TrimSpace(request.ID)

	if teamID == "" {
		return 0, fmt.Errorf("team id is required")
	}

	return apidata.RunWithTxid(ctx, database, func(transaction *sql.Tx) error {
		if err := assertTeamMembership(ctx, transaction, teamID, identity.Subject); err != nil {
			return err
		}

		assignments := []string{"updated_at = $1"}
		args := []any{time.Now().UTC()}
		position := 2

		appendString := func(column string, value *string, trim bool) {
			if value == nil {
				return
			}

			raw := *value

			if trim {
				raw = strings.TrimSpace(raw)
			}

			assignments = append(assignments, fmt.Sprintf("%s = $%d", column, position))
			args = append(args, raw)
			position++
		}

		appendString("name", request.Name, true)
		appendString("description", request.Description, true)
		appendString("color", request.Color, true)
		appendString("emoji", request.Emoji, true)

		if request.PrivacyMode != nil {
			mode := strings.TrimSpace(*request.PrivacyMode)

			if mode != "shared" && mode != "local" {
				return fmt.Errorf("privacy_mode must be 'shared' or 'local'")
			}

			assignments = append(assignments, fmt.Sprintf("privacy_mode = $%d", position))
			args = append(args, mode)
			position++
		}

		if len(assignments) == 1 {
			// Only updated_at — nothing meaningful to write.
			return nil
		}

		args = append(args, teamID)

		query := fmt.Sprintf(
			"UPDATE teams SET %s WHERE id = $%d",
			strings.Join(assignments, ", "),
			position,
		)

		if _, err := transaction.ExecContext(ctx.Context(), query, args...); err != nil {
			return fmt.Errorf("team update: %w", err)
		}

		return nil
	})
}

func assertTeamMembership(
	ctx fiber.Ctx,
	transaction *sql.Tx,
	teamID string,
	userID string,
) error {
	var exists bool

	err := transaction.QueryRowContext(
		ctx.Context(),
		`SELECT EXISTS (
      SELECT 1 FROM team_memberships
      WHERE team_id = $1 AND user_id = $2
    )`,
		teamID,
		userID,
	).Scan(&exists)

	if err != nil {
		return fmt.Errorf("team membership check: %w", err)
	}

	if !exists {
		return apidata.Forbidden(fmt.Errorf("not a member of this team"))
	}

	return nil
}
