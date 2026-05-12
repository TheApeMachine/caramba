package devteam

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"sync"

	"github.com/google/go-github/v67/github"
	_ "github.com/lib/pq"
	"golang.org/x/oauth2"

	"github.com/theapemachine/caramba/pkg/config"
)

/*
Orchestrator watches the kanban board for the "requests" project and manages
the full AI development team lifecycle for each card that enters the TODO column.

State machine per card:
  todo → (pick up) → in-progress → (developer loop + reviewer) → review → (PR opened)

The orchestrator is bounded by MaxConcurrent to avoid overwhelming the host.
*/
type Orchestrator struct {
	ctx     context.Context
	cancel  context.CancelFunc
	cfg     *config.DevTeamConfig
	db      *sql.DB
	github  *github.Client
	watcher *Watcher
	sem     chan struct{}
	wg      sync.WaitGroup
}

/*
NewOrchestrator constructs an Orchestrator from the loaded DevTeamConfig.
It does not start watching until Run is called.
*/
func NewOrchestrator(ctx context.Context, cfg *config.DevTeamConfig) (*Orchestrator, error) {
	db, err := sql.Open("postgres", cfg.DatabaseURL)

	if err != nil {
		return nil, fmt.Errorf("orchestrator: db: %w", err)
	}

	ts := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: cfg.GitHubToken})
	ghClient := github.NewClient(oauth2.NewClient(ctx, ts))

	ctx, cancel := context.WithCancel(ctx)

	return &Orchestrator{
		ctx:     ctx,
		cancel:  cancel,
		cfg:     cfg,
		db:      db,
		github:  ghClient,
		watcher: NewWatcher(ctx, cfg.DatabaseURL),
		sem:     make(chan struct{}, cfg.MaxConcurrent),
	}, nil
}

/*
Run starts the watcher goroutine and processes column-change events until ctx
is cancelled or Stop is called.
*/
func (orchestrator *Orchestrator) Run() error {
	orchestrator.wg.Add(1)

	go func() {
		defer orchestrator.wg.Done()
		_ = orchestrator.watcher.Watch()
	}()

	for {
		select {
		case <-orchestrator.ctx.Done():
			return nil

		case event := <-orchestrator.watcher.Events():
			if !orchestrator.isRelevant(event) {
				continue
			}

			orchestrator.sem <- struct{}{}
			orchestrator.wg.Add(1)

			go func(ev ColumnEvent) {
				defer func() {
					<-orchestrator.sem
					orchestrator.wg.Done()
				}()
				orchestrator.handle(ev)
			}(event)
		}
	}
}

/*
Stop cancels the orchestrator and waits for all in-flight goroutines to finish.
*/
func (orchestrator *Orchestrator) Stop() {
	orchestrator.cancel()
	orchestrator.wg.Wait()
	_ = orchestrator.db.Close()
}

// isRelevant filters events to only todo-bound cards on the requests project.
func (orchestrator *Orchestrator) isRelevant(event ColumnEvent) bool {
	return event.ColumnKey == "todo" &&
		event.ResearchProjectID == orchestrator.cfg.RequestsProjectID
}

/*
handle drives the full lifecycle for a single card:
  1. Move card to in-progress.
  2. Spin up a Docker sandbox.
  3. Developer agent implements the feature.
  4. Reviewer agent evaluates; loop back to developer on failure.
  5. On reviewer pass: commit, push, open PR, move card to review, destroy sandbox.
*/
func (orchestrator *Orchestrator) handle(event ColumnEvent) {
	branch := featureBranch(event)

	sandbox, err := NewSandbox(orchestrator.ctx, SandboxConfig{
		Image:         orchestrator.cfg.DockerImage,
		GitHubToken:   orchestrator.cfg.GitHubToken,
		GitHubOwner:   orchestrator.cfg.GitHubOwner,
		GitHubRepo:    orchestrator.cfg.GitHubRepo,
		FeatureBranch: branch,
	})

	if err != nil {
		orchestrator.moveCard(event.ID, "backlog", "sandbox creation failed: "+err.Error())
		return
	}

	defer sandbox.Destroy()

	orchestrator.moveCard(event.ID, "in-progress", "")

	if err := sandbox.Start(); err != nil {
		orchestrator.moveCard(event.ID, "backlog", "sandbox start failed: "+err.Error())
		return
	}

	developer := NewDeveloper(orchestrator.ctx, orchestrator.cfg.Developer, sandbox)
	reviewer := NewReviewer(orchestrator.ctx, orchestrator.cfg.Reviewer)

	feedback := ""

	for range maxIterations {
		if err := developer.Implement(event.Title, event.Description, feedback); err != nil {
			orchestrator.moveCard(event.ID, "backlog", "developer error: "+err.Error())
			return
		}

		verdict, err := reviewer.Review(sandbox, event.Title, event.Description)

		if err != nil {
			orchestrator.moveCard(event.ID, "backlog", "reviewer error: "+err.Error())
			return
		}

		if verdict.Pass {
			break
		}

		feedback = verdict.Feedback
	}

	commitMsg := fmt.Sprintf("feat: %s\n\n%s", event.Title, event.Description)

	if err := sandbox.CommitAndPush(commitMsg); err != nil {
		orchestrator.moveCard(event.ID, "backlog", "push failed: "+err.Error())
		return
	}

	prURL, err := orchestrator.openPR(event, branch)

	if err != nil {
		orchestrator.moveCard(event.ID, "backlog", "PR failed: "+err.Error())
		return
	}

	orchestrator.moveCard(event.ID, "review", prURL)
}

func (orchestrator *Orchestrator) moveCard(cardID, columnKey, note string) {
	q := `UPDATE kanban_cards SET column_key = $1, updated_at = NOW() WHERE id = $2`
	_, _ = orchestrator.db.ExecContext(orchestrator.ctx, q, columnKey, cardID)

	if note != "" {
		desc := fmt.Sprintf("[devteam] %s", note)
		_, _ = orchestrator.db.ExecContext(
			orchestrator.ctx,
			`UPDATE kanban_cards SET description = description || E'\n' || $1 WHERE id = $2`,
			desc, cardID,
		)
	}
}

func (orchestrator *Orchestrator) openPR(event ColumnEvent, branch string) (string, error) {
	base := "main"
	title := event.Title
	body := fmt.Sprintf(
		"Automated PR from the Caramba AI dev team.\n\nFeature request card: %s\n\n%s",
		event.ID, event.Description,
	)

	pr, _, err := orchestrator.github.PullRequests.Create(
		orchestrator.ctx,
		orchestrator.cfg.GitHubOwner,
		orchestrator.cfg.GitHubRepo,
		&github.NewPullRequest{
			Title: &title,
			Head:  &branch,
			Base:  &base,
			Body:  &body,
		},
	)

	if err != nil {
		return "", fmt.Errorf("orchestrator: open PR: %w", err)
	}

	return pr.GetHTMLURL(), nil
}

func featureBranch(event ColumnEvent) string {
	slug := strings.ToLower(event.Title)
	slug = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			return r
		}

		return '-'
	}, slug)

	parts := strings.FieldsFunc(slug, func(r rune) bool { return r == '-' })

	if len(parts) > 6 {
		parts = parts[:6]
	}

	return "devteam/" + event.ID[:8] + "-" + strings.Join(parts, "-")
}
