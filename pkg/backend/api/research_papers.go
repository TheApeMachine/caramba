package api

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/google/uuid"
)

/*
ResearchPaperService persists paper documents for research projects, with a
monotonic revision counter and append-only paper_revision_events for auditing
and future collaborative delta UIs.
*/
type ResearchPaperService struct {
	pool *sqlPool
}

type researchPaperCreateRequest struct {
	ID                string          `json:"id"`
	ResearchProjectID string          `json:"research_project_id"`
	Title             string          `json:"title"`
	Document          json.RawMessage `json:"document"`
}

type researchPaperUpdateRequest struct {
	ID               string          `json:"id"`
	ExpectedRevision int64           `json:"expected_revision"`
	Title            string          `json:"title"`
	Document         json.RawMessage `json:"document"`
	Patch            json.RawMessage `json:"patch"`
	Summary          string          `json:"summary"`
}

type revisionConflictError struct {
	ServerRevision int64 `json:"server_revision"`
}

func (revisionConflictError) Error() string {
	return "revision conflict: document was updated elsewhere"
}

func NewResearchPaperService(databaseURL string) *ResearchPaperService {
	return &ResearchPaperService{pool: newSQLPool(databaseURL)}
}

func (service *ResearchPaperService) Create(ctx fiber.Ctx) error {
	return mutate(ctx, "research paper", service.create)
}

func (service *ResearchPaperService) create(
	ctx fiber.Ctx,
	identity clerkIdentity,
	request researchPaperCreateRequest,
) (int64, error) {
	database, err := service.pool.open()

	if err != nil {
		return 0, err
	}

	orgSlug := strings.TrimSpace(identity.orgSlug)

	if orgSlug == "" {
		orgSlug = strings.TrimSpace(identity.subject)
	}

	if orgSlug == "" {
		return 0, forbidden(errors.New("organization scope required for paper create"))
	}

	projectID := strings.TrimSpace(request.ResearchProjectID)

	if projectID == "" {
		return 0, fmt.Errorf("research_project_id is required")
	}

	paperID := strings.TrimSpace(request.ID)

	if paperID == "" {
		return 0, fmt.Errorf("paper id is required")
	}

	document := request.Document

	trimmed := strings.TrimSpace(string(document))

	if len(document) == 0 || trimmed == "" || trimmed == "{}" || trimmed == "null" {
		document = service.defaultDocumentJSON()
	}

	if !json.Valid(document) {
		return 0, fmt.Errorf("document must be valid JSON")
	}

	title := strings.TrimSpace(request.Title)

	if title == "" {
		title = extractTitleFromDocument(document, "Untitled paper")
	}

	return runWithTxid(ctx, database, func(transaction *sql.Tx) error {
		if err := service.requireProjectInOrganization(transaction, ctx, projectID, orgSlug); err != nil {
			return err
		}

		now := time.Now().UTC()

		_, err := transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO research_papers (
            id,
            research_project_id,
            organization_slug,
            title,
            document,
            revision,
            created_at,
            updated_at
          ) VALUES ($1, $2, $3, $4, $5::jsonb, 1, $6, $7)`,
			paperID,
			projectID,
			orgSlug,
			title,
			document,
			now,
			now,
		)

		if err != nil {
			return fmt.Errorf("research paper insert: %w", err)
		}

		eventID := uuid.New().String()
		subject := strings.TrimSpace(identity.subject)

		_, err = transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO paper_revision_events (
            id,
            paper_id,
            revision,
            previous_revision,
            document,
            patch,
            summary,
            actor_id,
            created_at
          ) VALUES ($1, $2, $3, $4, $5::jsonb, NULL, $6, $7, $8)`,
			eventID,
			paperID,
			int64(1),
			int64(0),
			document,
			"create",
			subject,
			now,
		)

		if err != nil {
			return fmt.Errorf("paper revision event insert: %w", err)
		}

		return nil
	})
}

func (service *ResearchPaperService) Update(ctx fiber.Ctx) error {
	request := researchPaperUpdateRequest{}

	if err := ctx.Bind().JSON(&request); err != nil {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid research paper update payload"})
	}

	identity, err := readClerkIdentity(ctx)

	if err != nil {
		return ctx.Status(fiber.StatusUnauthorized).JSON(fiber.Map{"error": err.Error()})
	}

	database, err := service.pool.open()

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	orgSlug := strings.TrimSpace(identity.orgSlug)

	if orgSlug == "" {
		orgSlug = strings.TrimSpace(identity.subject)
	}

	paperID := strings.TrimSpace(request.ID)

	if paperID == "" {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "paper id is required"})
	}

	if request.ExpectedRevision < 1 {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "expected_revision must be at least 1"})
	}

	document := request.Document

	if len(document) == 0 || !json.Valid(document) {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "document must be valid JSON"})
	}

	title := strings.TrimSpace(request.Title)

	if title == "" {
		title = extractTitleFromDocument(document, "")
	}

	patch := request.Patch

	if len(patch) == 0 || string(patch) == "null" {
		patch = nil
	}

	summary := strings.TrimSpace(request.Summary)

	if summary == "" {
		summary = "update"
	}

	subject := strings.TrimSpace(identity.subject)

	txid, updateErr := runWithTxid(ctx, database, func(transaction *sql.Tx) error {
		now := time.Now().UTC()

		var nextRevision int64

		row := transaction.QueryRowContext(
			ctx.Context(),
			`UPDATE research_papers
        SET document = $1::jsonb,
            title = $2,
            revision = revision + 1,
            updated_at = $3
        WHERE id = $4
          AND organization_slug = $5
          AND revision = $6
        RETURNING revision`,
			document,
			title,
			now,
			paperID,
			orgSlug,
			request.ExpectedRevision,
		)

		if err := row.Scan(&nextRevision); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				current := service.queryCurrentRevision(transaction, ctx, paperID, orgSlug)

				return revisionConflictError{ServerRevision: current}
			}

			return fmt.Errorf("research paper update: %w", err)
		}

		eventID := uuid.New().String()

		var patchValue any

		if len(patch) > 0 {
			patchValue = patch
		}

		_, err := transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO paper_revision_events (
          id,
          paper_id,
          revision,
          previous_revision,
          document,
          patch,
          summary,
          actor_id,
          created_at
        ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9)`,
			eventID,
			paperID,
			nextRevision,
			request.ExpectedRevision,
			document,
			patchValue,
			summary,
			subject,
			now,
		)

		if err != nil {
			return fmt.Errorf("paper revision event insert: %w", err)
		}

		return nil
	})

	if updateErr != nil {
		var conflict revisionConflictError

		if errors.As(updateErr, &conflict) {
			return ctx.Status(fiber.StatusConflict).JSON(fiber.Map{
				"error":    conflict.Error(),
				"revision": conflict.ServerRevision,
			})
		}

		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": updateErr.Error()})
	}

	return ctx.JSON(fiber.Map{"txid": txid})
}

func (service *ResearchPaperService) requireProjectInOrganization(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	projectID string,
	orgSlug string,
) error {
	var rowOrg string

	err := transaction.QueryRowContext(
		ctx.Context(),
		`SELECT organization_slug FROM research_projects WHERE id = $1`,
		projectID,
	).Scan(&rowOrg)

	if errors.Is(err, sql.ErrNoRows) {
		return fmt.Errorf("research project not found")
	}

	if err != nil {
		return fmt.Errorf("research project lookup: %w", err)
	}

	if strings.TrimSpace(rowOrg) != orgSlug {
		return forbidden(errors.New("project is not in this organization"))
	}

	return nil
}

func (service *ResearchPaperService) queryCurrentRevision(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	paperID string,
	orgSlug string,
) int64 {
	var rev sql.NullInt64

	_ = transaction.QueryRowContext(
		ctx.Context(),
		`SELECT revision FROM research_papers WHERE id = $1 AND organization_slug = $2`,
		paperID,
		orgSlug,
	).Scan(&rev)

	if rev.Valid {
		return rev.Int64
	}

	return 0
}

func (service *ResearchPaperService) defaultDocumentJSON() json.RawMessage {
	headingID := uuid.New().String()
	paragraphID := uuid.New().String()

	raw := fmt.Sprintf(
		`{"metadata":{"title":"","authors":"","keywords":"","abstract":""},"blocks":[`+
			`{"id":%q,"type":"heading","level":1,"text":"Untitled paper"},`+
			`{"id":%q,"type":"paragraph","text":""}]}`,
		headingID,
		paragraphID,
	)

	return json.RawMessage(raw)
}

func extractTitleFromDocument(document json.RawMessage, fallback string) string {
	var envelope struct {
		Metadata struct {
			Title string `json:"title"`
		} `json:"metadata"`
	}

	if err := json.Unmarshal(document, &envelope); err != nil {
		return fallback
	}

	title := strings.TrimSpace(envelope.Metadata.Title)

	if title != "" {
		return title
	}

	return fallback
}
