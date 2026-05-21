package research

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"
	"unicode"

	"github.com/gofiber/fiber/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/backend/apidata"
)

type ProjectService struct {
	pool *apidata.SQLPool
}

type createRequest struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	ProjectSlug string `json:"project_slug"`
}

type provisionPaperRequest struct {
	ID    string `json:"id"`
	Title string `json:"title"`
}

type provisionRequest struct {
	ID          string                  `json:"id"`
	Name        string                  `json:"name"`
	Description string                  `json:"description"`
	ProjectSlug string                  `json:"project_slug"`
	PaperID     string                  `json:"paper_id"`
	Papers      []provisionPaperRequest `json:"papers"`
	MemberIDs   []string                `json:"member_ids"`
}

type starterKanbanCard struct {
	Title       string
	Description string
	ColumnKey   string
}

var starterCards = []starterKanbanCard{
	{
		Title:       "Project kickoff",
		Description: "Align on goals, scope, and success criteria for this research effort.",
		ColumnKey:   "todo",
	},
	{
		Title:       "Draft architecture",
		Description: "Sketch the model graph, data flow, and planned ablations.",
		ColumnKey:   "backlog",
	},
	{
		Title:       "Paper outlines",
		Description: "Draft one or more papers tied to this project as results mature.",
		ColumnKey:   "backlog",
	},
}

func NewProjectService(databaseURL string) *ProjectService {
	return &ProjectService{pool: apidata.NewSQLPool(databaseURL)}
}

func (service *ProjectService) Create(ctx fiber.Ctx) error {
	return apidata.Mutate(ctx, "research project", service.create)
}

func (service *ProjectService) Provision(ctx fiber.Ctx) error {
	return apidata.Mutate(ctx, "research project provision", service.provision)
}

func (service *ProjectService) create(
	ctx fiber.Ctx,
	identity apidata.ClerkIdentity,
	request createRequest,
) (int64, error) {
	database, err := service.pool.Open()

	if err != nil {
		return 0, err
	}

	orgSlug := organizationSlugFromIdentity(identity)

	return apidata.RunWithTxid(ctx, database, func(transaction *sql.Tx) error {
		return service.insertProjectRow(
			transaction,
			ctx,
			orgSlug,
			request.ID,
			request.Name,
			request.Description,
			strings.TrimSpace(request.ProjectSlug),
		)
	})
}

func (service *ProjectService) provision(
	ctx fiber.Ctx,
	identity apidata.ClerkIdentity,
	request provisionRequest,
) (int64, error) {
	database, err := service.pool.Open()

	if err != nil {
		return 0, err
	}

	orgSlug := organizationSlugFromIdentity(identity)
	projectID := strings.TrimSpace(request.ID)

	if projectID == "" {
		return 0, fmt.Errorf("research project id is required")
	}

	papers := normalizeProvisionPapers(request)
	memberIDs := dedupeMemberIDs(identity.Subject, request.MemberIDs)

	return apidata.RunWithTxid(ctx, database, func(transaction *sql.Tx) error {
		projectSlug, err := service.resolveProjectSlug(
			transaction,
			ctx,
			orgSlug,
			strings.TrimSpace(request.ProjectSlug),
			request.Name,
		)

		if err != nil {
			return err
		}

		if err := service.insertProjectRow(
			transaction,
			ctx,
			orgSlug,
			projectID,
			request.Name,
			request.Description,
			projectSlug,
		); err != nil {
			return err
		}

		if err := service.insertProjectMembers(
			transaction,
			ctx,
			projectID,
			identity.Subject,
			memberIDs,
		); err != nil {
			return err
		}

		if err := service.insertStarterKanbanCards(
			transaction,
			ctx,
			projectID,
			identity.Subject,
		); err != nil {
			return err
		}

		return service.insertProvisionPapers(
			transaction,
			ctx,
			projectID,
			orgSlug,
			papers,
			identity.Subject,
		)
	})
}

func normalizeProvisionPapers(request provisionRequest) []provisionPaperRequest {
	papers := request.Papers

	if len(papers) == 0 {
		legacyPaperID := strings.TrimSpace(request.PaperID)

		if legacyPaperID != "" {
			return []provisionPaperRequest{{
				ID:    legacyPaperID,
				Title: strings.TrimSpace(request.Name),
			}}
		}

		return nil
	}

	normalized := make([]provisionPaperRequest, 0, len(papers))

	for _, paper := range papers {
		paperID := strings.TrimSpace(paper.ID)

		if paperID == "" {
			continue
		}

		normalized = append(normalized, provisionPaperRequest{
			ID:    paperID,
			Title: strings.TrimSpace(paper.Title),
		})
	}

	return normalized
}

func organizationSlugFromIdentity(identity apidata.ClerkIdentity) string {
	orgSlug := strings.TrimSpace(identity.OrgSlug)

	if orgSlug == "" {
		orgSlug = strings.TrimSpace(identity.Subject)
	}

	return orgSlug
}

func dedupeMemberIDs(ownerID string, memberIDs []string) []string {
	seen := map[string]struct{}{}
	result := make([]string, 0, len(memberIDs)+1)

	appendUnique := func(userID string) {
		userID = strings.TrimSpace(userID)

		if userID == "" {
			return
		}

		if _, ok := seen[userID]; ok {
			return
		}

		seen[userID] = struct{}{}
		result = append(result, userID)
	}

	appendUnique(ownerID)

	for _, memberID := range memberIDs {
		appendUnique(memberID)
	}

	return result
}

func (service *ProjectService) insertProjectRow(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	organizationSlug string,
	projectID string,
	name string,
	description string,
	projectSlug string,
) error {
	name = strings.TrimSpace(name)

	if name == "" {
		return fmt.Errorf("research project name is required")
	}

	now := time.Now().UTC()

	_, err := transaction.ExecContext(
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
		projectID,
		name,
		strings.TrimSpace(description),
		organizationSlug,
		apidata.NullString(projectSlug),
		now,
		now,
	)

	if err != nil {
		return fmt.Errorf("research project insert: %w", err)
	}

	return nil
}

func (service *ProjectService) resolveProjectSlug(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	organizationSlug string,
	requestedSlug string,
	projectName string,
) (string, error) {
	base := deriveProjectSlug(projectName)

	if requestedSlug != "" {
		base = deriveProjectSlug(requestedSlug)
	}

	return service.ensureUniqueProjectSlug(transaction, ctx, organizationSlug, base)
}

func (service *ProjectService) ensureUniqueProjectSlug(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	organizationSlug string,
	base string,
) (string, error) {
	candidate := base

	for suffix := 0; suffix < 100; suffix++ {
		if suffix > 0 {
			candidate = fmt.Sprintf("%s-%d", base, suffix+1)
		}

		var exists bool

		err := transaction.QueryRowContext(
			ctx.Context(),
			`SELECT EXISTS (
          SELECT 1 FROM research_projects
          WHERE organization_slug = $1 AND project_slug = $2
        )`,
			organizationSlug,
			candidate,
		).Scan(&exists)

		if err != nil {
			return "", fmt.Errorf("research project slug lookup: %w", err)
		}

		if !exists {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("could not allocate a unique project slug")
}

func (service *ProjectService) insertProjectMembers(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	projectID string,
	ownerID string,
	memberIDs []string,
) error {
	now := time.Now().UTC()

	for _, memberID := range memberIDs {
		role := "member"

		if memberID == ownerID {
			role = "owner"
		}

		_, err := transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO research_project_members (
          research_project_id,
          user_id,
          role,
          created_at
        ) VALUES ($1, $2, $3, $4)
        ON CONFLICT (research_project_id, user_id) DO NOTHING`,
			projectID,
			memberID,
			role,
			now,
		)

		if err != nil {
			return fmt.Errorf("research project member insert: %w", err)
		}
	}

	return nil
}

func (service *ProjectService) insertStarterKanbanCards(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	projectID string,
	ownerID string,
) error {
	assigneesJSON, err := json.Marshal([]string{ownerID})

	if err != nil {
		return fmt.Errorf("starter card assignees json: %w", err)
	}

	now := time.Now().UTC()

	for index, starter := range starterCards {
		_, err := transaction.ExecContext(
			ctx.Context(),
			`INSERT INTO kanban_cards (
          id,
          research_project_id,
          column_key,
          sort_order,
          title,
          description,
          priority,
          labels_json,
          assignees_json,
          due_date,
          requested_by,
          created_at,
          updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NULL, NULL, $10, $11)`,
			uuid.New().String(),
			projectID,
			starter.ColumnKey,
			index,
			starter.Title,
			starter.Description,
			"medium",
			"[]",
			string(assigneesJSON),
			now,
			now,
		)

		if err != nil {
			return fmt.Errorf("starter kanban card insert: %w", err)
		}
	}

	return nil
}

func (service *ProjectService) insertProvisionPapers(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	projectID string,
	organizationSlug string,
	papers []provisionPaperRequest,
	actorID string,
) error {
	for _, paper := range papers {
		if err := service.insertProvisionPaper(
			transaction,
			ctx,
			paper.ID,
			projectID,
			organizationSlug,
			paper.Title,
			actorID,
		); err != nil {
			return err
		}
	}

	return nil
}

func (service *ProjectService) insertProvisionPaper(
	transaction *sql.Tx,
	ctx fiber.Ctx,
	paperID string,
	projectID string,
	organizationSlug string,
	paperTitle string,
	actorID string,
) error {
	document := defaultPaperDocumentJSON()
	title := strings.TrimSpace(paperTitle)

	if title == "" {
		title = "Untitled paper"
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
		organizationSlug,
		title,
		document,
		now,
		now,
	)

	if err != nil {
		return fmt.Errorf("research paper insert: %w", err)
	}

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
		uuid.New().String(),
		paperID,
		int64(1),
		int64(0),
		document,
		"provision",
		strings.TrimSpace(actorID),
		now,
	)

	if err != nil {
		return fmt.Errorf("paper revision event insert: %w", err)
	}

	return nil
}

func defaultPaperDocumentJSON() json.RawMessage {
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

/*
deriveProjectSlug turns a display name into a URL-safe slug used for Kanban routing.
*/
func deriveProjectSlug(name string) string {
	var builder strings.Builder
	lastDash := false

	for _, character := range strings.ToLower(strings.TrimSpace(name)) {
		if unicode.IsLetter(character) || unicode.IsDigit(character) {
			builder.WriteRune(character)
			lastDash = false

			continue
		}

		if !lastDash && builder.Len() > 0 {
			builder.WriteByte('-')
			lastDash = true
		}
	}

	slug := strings.Trim(builder.String(), "-")

	if slug == "" {
		return "project"
	}

	const maxSlugLength = 64

	if len(slug) > maxSlugLength {
		slug = strings.TrimRight(slug[:maxSlugLength], "-")
	}

	return slug
}
