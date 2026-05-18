package api

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/gofiber/fiber/v3"
)

type ResearcherProfileService struct {
	pool *sqlPool
}

type researcherProfilePayload struct {
	DisplayName   string `json:"display_name"`
	RoleTitle     string `json:"role_title"`
	Affiliation   string `json:"affiliation"`
	Bio           string `json:"bio"`
	Website       string `json:"website"`
	ResearchFocus string `json:"research_focus"`
}

type researcherProfileResponse struct {
	UserID        string `json:"user_id"`
	DisplayName   string `json:"display_name"`
	RoleTitle     string `json:"role_title"`
	Affiliation   string `json:"affiliation"`
	Bio           string `json:"bio"`
	Website       string `json:"website"`
	ResearchFocus string `json:"research_focus"`
	UpdatedAt     string `json:"updated_at"`
}

func NewResearcherProfileService(databaseURL string) *ResearcherProfileService {
	return &ResearcherProfileService{pool: newSQLPool(databaseURL)}
}

func (service *ResearcherProfileService) Get(ctx fiber.Ctx) error {
	identity, err := readClerkIdentity(ctx)

	if err != nil {
		return ctx.Status(fiber.StatusUnauthorized).JSON(fiber.Map{"error": err.Error()})
	}

	database, err := service.pool.open()

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	profile, err := service.loadProfile(ctx, database, identity.subject)

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	return ctx.JSON(profile)
}

func (service *ResearcherProfileService) Save(ctx fiber.Ctx) error {
	return mutate(ctx, "researcher profile", service.save)
}

func (service *ResearcherProfileService) save(
	ctx fiber.Ctx,
	identity clerkIdentity,
	request researcherProfilePayload,
) (int64, error) {
	database, err := service.pool.open()

	if err != nil {
		return 0, err
	}

	now := time.Now().UTC()

	return runWithTxid(ctx, database, func(transaction *sql.Tx) error {
		_, err := transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO researcher_profiles (
          user_id,
          display_name,
          role_title,
          affiliation,
          bio,
          website,
          research_focus,
          created_at,
          updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (user_id) DO UPDATE SET
          display_name = EXCLUDED.display_name,
          role_title = EXCLUDED.role_title,
          affiliation = EXCLUDED.affiliation,
          bio = EXCLUDED.bio,
          website = EXCLUDED.website,
          research_focus = EXCLUDED.research_focus,
          updated_at = EXCLUDED.updated_at`,
			identity.subject,
			strings.TrimSpace(request.DisplayName),
			strings.TrimSpace(request.RoleTitle),
			strings.TrimSpace(request.Affiliation),
			strings.TrimSpace(request.Bio),
			strings.TrimSpace(request.Website),
			strings.TrimSpace(request.ResearchFocus),
			now,
			now,
		)

		if err != nil {
			return fmt.Errorf("researcher profile upsert: %w", err)
		}

		return nil
	})
}

func (service *ResearcherProfileService) loadProfile(
	ctx fiber.Ctx,
	database *sql.DB,
	userID string,
) (researcherProfileResponse, error) {
	row := database.QueryRowContext(
		ctx.Context(),
		`SELECT
          user_id,
          display_name,
          role_title,
          affiliation,
          bio,
          website,
          research_focus,
          updated_at
        FROM researcher_profiles
        WHERE user_id = $1`,
		userID,
	)

	var profile researcherProfileResponse
	var updatedAt time.Time

	err := row.Scan(
		&profile.UserID,
		&profile.DisplayName,
		&profile.RoleTitle,
		&profile.Affiliation,
		&profile.Bio,
		&profile.Website,
		&profile.ResearchFocus,
		&updatedAt,
	)

	if errors.Is(err, sql.ErrNoRows) {
		return researcherProfileResponse{UserID: userID}, nil
	}

	if err != nil {
		return researcherProfileResponse{}, fmt.Errorf("researcher profile load: %w", err)
	}

	profile.UpdatedAt = updatedAt.UTC().Format(time.RFC3339)

	return profile, nil
}
