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
}

type teamListRow struct {
	ID               string    `json:"id"`
	OrganizationSlug string    `json:"organization_slug"`
	Name             string    `json:"name"`
	Slug             string    `json:"slug"`
	Description      string    `json:"description"`
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

		_, err = transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO teams (id, organization_slug, name, slug, description, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $6)`,
			teamID,
			orgSlug,
			name,
			slug,
			strings.TrimSpace(request.Description),
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
