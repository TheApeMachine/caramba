package api

import (
	"database/sql"
	"fmt"
	"strings"
	"time"

	"github.com/gofiber/fiber/v3"
	_ "github.com/lib/pq"
)

type ResearchProjectService struct {
	databaseURL string
	database    *sql.DB
}

type researchProjectCreateRequest struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	ProjectSlug string `json:"project_slug"`
}

func NewResearchProjectService(databaseURL string) *ResearchProjectService {
	return &ResearchProjectService{databaseURL: strings.TrimSpace(databaseURL)}
}

func (service *ResearchProjectService) Create(ctx fiber.Ctx) error {
	request := researchProjectCreateRequest{}

	if err := ctx.Bind().JSON(&request); err != nil {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid research project payload"})
	}

	organizationSlug, _ := ctx.Locals("clerkOrganizationSlug").(string)

	if strings.TrimSpace(organizationSlug) == "" {
		subject, _ := ctx.Locals("clerkSubject").(string)
		organizationSlug = strings.TrimSpace(subject)
	}

	if organizationSlug == "" {
		return ctx.Status(fiber.StatusForbidden).JSON(fiber.Map{"error": "authenticated identity required"})
	}

	txid, err := service.insert(ctx, organizationSlug, request)

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	return ctx.JSON(fiber.Map{"txid": txid})
}

func (service *ResearchProjectService) insert(
	ctx fiber.Ctx,
	organizationSlug string,
	request researchProjectCreateRequest,
) (int64, error) {
	database, err := service.open()

	if err != nil {
		return 0, err
	}

	name := strings.TrimSpace(request.Name)

	if name == "" {
		return 0, fmt.Errorf("research project name is required")
	}

	transaction, err := database.BeginTx(ctx.Context(), nil)

	if err != nil {
		return 0, fmt.Errorf("research project insert begin: %w", err)
	}

	defer transaction.Rollback()

	now := time.Now().UTC()

	_, err = transaction.ExecContext(
		ctx.Context(),
		`INSERT INTO research_projects (
        id,
        name,
        description,
        organization_slug,
        project_slug,
        created_at,
        updated_at
      )
       VALUES ($1, $2, $3, $4, $5, $6, $7)`,
		request.ID,
		name,
		strings.TrimSpace(request.Description),
		strings.TrimSpace(organizationSlug),
		nullString(strings.TrimSpace(request.ProjectSlug)),
		now,
		now,
	)

	if err != nil {
		return 0, fmt.Errorf("research project insert: %w", err)
	}

	txidRow := transaction.QueryRowContext(
		ctx.Context(),
		"SELECT pg_current_xact_id()::xid::text AS txid",
	)
	var txidRaw string

	if err := txidRow.Scan(&txidRaw); err != nil {
		return 0, fmt.Errorf("research project insert txid: %w", err)
	}

	if err := transaction.Commit(); err != nil {
		return 0, fmt.Errorf("research project insert commit: %w", err)
	}

	var txid int64

	if _, err := fmt.Sscan(txidRaw, &txid); err != nil {
		return 0, fmt.Errorf("research project insert txid parse: %w", err)
	}

	return txid, nil
}

func (service *ResearchProjectService) open() (*sql.DB, error) {
	if service.database != nil {
		return service.database, nil
	}

	if service.databaseURL == "" {
		return nil, fmt.Errorf("research project database_url is required")
	}

	database, err := sql.Open("postgres", service.databaseURL)

	if err != nil {
		return nil, fmt.Errorf("research project database open: %w", err)
	}

	service.database = database

	return service.database, nil
}

func nullString(value string) sql.NullString {
	return sql.NullString{String: value, Valid: value != ""}
}
