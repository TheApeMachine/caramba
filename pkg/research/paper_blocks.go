package research

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/backend/apidata"
)

/*
PaperBlockService persists the individual blocks (paragraph / heading /
equation / list) that make up a research paper. Mutations flow row-by-row
through TanStack DB collection.insert / update / delete on the client, hit
these endpoints, and Electric syncs the writes back to every subscriber.
*/
type PaperBlockService struct {
	pool *apidata.SQLPool
}

type paperBlockUpsertRequest struct {
	ID                  string `json:"id"`
	PaperID             string `json:"paper_id"`
	SortOrder           int64  `json:"sort_order"`
	Kind                string `json:"kind"`
	Text                string `json:"text"`
	Latex               string `json:"latex"`
	HeadingLevel        *int16 `json:"heading_level"`
	HeadingPresentation string `json:"heading_presentation"`
	ListOrdered         bool   `json:"list_ordered"`
	EquationDisplay     bool   `json:"equation_display"`
	EquationLabel       string `json:"equation_label"`
}

type paperBlockDeleteRequest struct {
	ID string `json:"id"`
}

type paperBlockReorderEntry struct {
	ID        string `json:"id"`
	SortOrder int64  `json:"sort_order"`
}

type paperBlockReorderRequest struct {
	PaperID string                   `json:"paper_id"`
	Entries []paperBlockReorderEntry `json:"entries"`
}

var allowedBlockKinds = map[string]struct{}{
	"paragraph": {},
	"heading":   {},
	"equation":  {},
	"list":      {},
}

var allowedHeadingPresentations = map[string]struct{}{
	"abstract":        {},
	"references":      {},
	"acknowledgments": {},
}

func NewPaperBlockService(databaseURL string) *PaperBlockService {
	return &PaperBlockService{pool: apidata.NewSQLPool(databaseURL)}
}

func (service *PaperBlockService) Create(ctx fiber.Ctx) error {
	return apidata.Mutate(ctx, "research paper block", service.upsert)
}

func (service *PaperBlockService) Update(ctx fiber.Ctx) error {
	return apidata.Mutate(ctx, "research paper block", service.upsert)
}

func (service *PaperBlockService) Delete(ctx fiber.Ctx) error {
	return apidata.Mutate(ctx, "research paper block", service.delete)
}

func (service *PaperBlockService) Reorder(ctx fiber.Ctx) error {
	return apidata.Mutate(ctx, "research paper block reorder", service.reorder)
}

func (service *PaperBlockService) upsert(
	ctx fiber.Ctx,
	identity apidata.ClerkIdentity,
	request paperBlockUpsertRequest,
) (int64, error) {
	database, err := service.pool.Open()

	if err != nil {
		return 0, err
	}

	orgSlug := organizationSlugFromIdentity(identity)
	blockID := strings.TrimSpace(request.ID)
	paperID := strings.TrimSpace(request.PaperID)

	if blockID == "" {
		return 0, fmt.Errorf("block id is required")
	}

	if paperID == "" {
		return 0, fmt.Errorf("paper_id is required")
	}

	kind := strings.TrimSpace(request.Kind)

	if _, ok := allowedBlockKinds[kind]; !ok {
		return 0, fmt.Errorf("unsupported block kind %q", kind)
	}

	if kind == "heading" {
		if request.HeadingLevel == nil {
			return 0, fmt.Errorf("heading_level is required for heading blocks")
		}

		if *request.HeadingLevel < 1 || *request.HeadingLevel > 3 {
			return 0, fmt.Errorf("heading_level must be between 1 and 3")
		}
	}

	presentation := strings.TrimSpace(request.HeadingPresentation)

	if presentation != "" {
		if _, ok := allowedHeadingPresentations[presentation]; !ok {
			return 0, fmt.Errorf("unsupported heading_presentation %q", presentation)
		}
	}

	return apidata.RunWithTxid(ctx, database, func(transaction *sql.Tx) error {
		if err := service.requirePaperInOrganization(transaction, ctx, paperID, orgSlug); err != nil {
			return err
		}

		now := time.Now().UTC()
		headingLevel := nullableHeadingLevel(kind, request.HeadingLevel)
		presentationValue := apidata.NullString(presentation)

		_, err := transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO research_paper_blocks (
          id,
          paper_id,
          organization_slug,
          sort_order,
          kind,
          text,
          latex,
          heading_level,
          heading_presentation,
          list_ordered,
          equation_display,
          equation_label,
          created_at,
          updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $13)
        ON CONFLICT (id) DO UPDATE SET
          paper_id = EXCLUDED.paper_id,
          organization_slug = EXCLUDED.organization_slug,
          sort_order = EXCLUDED.sort_order,
          kind = EXCLUDED.kind,
          text = EXCLUDED.text,
          latex = EXCLUDED.latex,
          heading_level = EXCLUDED.heading_level,
          heading_presentation = EXCLUDED.heading_presentation,
          list_ordered = EXCLUDED.list_ordered,
          equation_display = EXCLUDED.equation_display,
          equation_label = EXCLUDED.equation_label,
          updated_at = EXCLUDED.updated_at`,
			blockID,
			paperID,
			orgSlug,
			request.SortOrder,
			kind,
			request.Text,
			request.Latex,
			headingLevel,
			presentationValue,
			request.ListOrdered,
			request.EquationDisplay,
			request.EquationLabel,
			now,
		)

		if err != nil {
			return fmt.Errorf("research paper block upsert: %w", err)
		}

		return nil
	})
}

func (service *PaperBlockService) delete(
	ctx fiber.Ctx,
	identity apidata.ClerkIdentity,
	request paperBlockDeleteRequest,
) (int64, error) {
	database, err := service.pool.Open()

	if err != nil {
		return 0, err
	}

	orgSlug := organizationSlugFromIdentity(identity)
	blockID := strings.TrimSpace(request.ID)

	if blockID == "" {
		return 0, fmt.Errorf("block id is required")
	}

	return apidata.RunWithTxid(ctx, database, func(transaction *sql.Tx) error {
		result, err := transaction.ExecContext(
			ctx.Context(),
			`DELETE FROM research_paper_blocks
        WHERE id = $1 AND organization_slug = $2`,
			blockID,
			orgSlug,
		)

		if err != nil {
			return fmt.Errorf("research paper block delete: %w", err)
		}

		affected, err := result.RowsAffected()

		if err != nil {
			return fmt.Errorf("research paper block delete rows: %w", err)
		}

		if affected == 0 {
			return apidata.Forbidden(errors.New("block not found in this organization"))
		}

		return nil
	})
}

func (service *PaperBlockService) reorder(
	ctx fiber.Ctx,
	identity apidata.ClerkIdentity,
	request paperBlockReorderRequest,
) (int64, error) {
	database, err := service.pool.Open()

	if err != nil {
		return 0, err
	}

	orgSlug := organizationSlugFromIdentity(identity)
	paperID := strings.TrimSpace(request.PaperID)

	if paperID == "" {
		return 0, fmt.Errorf("paper_id is required")
	}

	if len(request.Entries) == 0 {
		return 0, fmt.Errorf("reorder entries are required")
	}

	return apidata.RunWithTxid(ctx, database, func(transaction *sql.Tx) error {
		if err := service.requirePaperInOrganization(transaction, ctx, paperID, orgSlug); err != nil {
			return err
		}

		now := time.Now().UTC()

		for _, entry := range request.Entries {
			entryID := strings.TrimSpace(entry.ID)

			if entryID == "" {
				return fmt.Errorf("reorder entry id is required")
			}

			_, err := transaction.ExecContext(
				ctx.Context(),
				`UPDATE research_paper_blocks
            SET sort_order = $1,
                updated_at = $2
          WHERE id = $3
            AND paper_id = $4
            AND organization_slug = $5`,
				entry.SortOrder,
				now,
				entryID,
				paperID,
				orgSlug,
			)

			if err != nil {
				return fmt.Errorf("research paper block reorder %s: %w", entryID, err)
			}
		}

		return nil
	})
}

func (service *PaperBlockService) requirePaperInOrganization(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	paperID string,
	orgSlug string,
) error {
	var rowOrg string

	err := transaction.QueryRowContext(
		ctx.Context(),
		`SELECT organization_slug FROM research_papers WHERE id = $1`,
		paperID,
	).Scan(&rowOrg)

	if errors.Is(err, sql.ErrNoRows) {
		return fmt.Errorf("research paper not found")
	}

	if err != nil {
		return fmt.Errorf("research paper lookup: %w", err)
	}

	if strings.TrimSpace(rowOrg) != orgSlug {
		return apidata.Forbidden(errors.New("paper is not in this organization"))
	}

	return nil
}

func nullableHeadingLevel(kind string, level *int16) sql.NullInt16 {
	if kind != "heading" || level == nil {
		return sql.NullInt16{}
	}

	return sql.NullInt16{Int16: *level, Valid: true}
}
