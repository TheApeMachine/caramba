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

Flow per card:
  todo → Planner decomposes into subtasks → subtasks fan out to Developer agents
       → each subtask runs its own developer→reviewer loop in its own sandbox
       → card moves to review only when all subtasks reach done

A shared FileLockRegistry prevents concurrent developer agents from
unknowingly generating conflicting changes to the same logical file path.
*/
type Orchestrator struct {
	ctx       context.Context
	cancel    context.CancelFunc
	cfg       *config.DevTeamConfig
	db        *sql.DB
	subtasks  *SubtaskStore
	github    *github.Client
	watcher   *Watcher
	locks     *FileLockRegistry
	extractor *ContextExtractor
	sem       chan struct{}
	wg        sync.WaitGroup
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

	extractor, err := NewContextExtractor()

	if err != nil {
		return nil, fmt.Errorf("orchestrator: context extractor: %w", err)
	}

	ctx, cancel := context.WithCancel(ctx)

	return &Orchestrator{
		ctx:       ctx,
		cancel:    cancel,
		cfg:       cfg,
		db:        db,
		subtasks:  NewSubtaskStore(db),
		github:    ghClient,
		watcher:   NewWatcher(ctx, cfg.DatabaseURL),
		locks:     NewFileLockRegistry(ctx),
		extractor: extractor,
		sem:       make(chan struct{}, cfg.MaxConcurrent),
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
	orchestrator.locks.Close()
	orchestrator.extractor.Close()
	_ = orchestrator.db.Close()
}

func (orchestrator *Orchestrator) isRelevant(event ColumnEvent) bool {
	return event.ColumnKey == "todo" &&
		event.ResearchProjectID == orchestrator.cfg.RequestsProjectID
}

/*
handle is the top-level card lifecycle:
 1. Extract blast radius from the host repo.
 2. Spin up a planner sandbox, run the Planner agent, persist subtasks.
 3. Fan out: each subtask gets its own sandbox and developer→reviewer loop.
 4. Wait for all subtasks; move card to review (or backlog on failure).
*/
func (orchestrator *Orchestrator) handle(event ColumnEvent) {
	orchestrator.moveCard(event.ID, "in-progress", "")

	blastContext := orchestrator.extractContext(event)

	subtaskDrafts, err := orchestrator.runPlanner(event, blastContext)

	if err != nil {
		orchestrator.moveCard(event.ID, "backlog", "planner error: "+err.Error())
		return
	}

	if len(subtaskDrafts) == 0 {
		orchestrator.moveCard(event.ID, "backlog", "planner produced no subtasks")
		return
	}

	branch := featureBranch(event)

	subtaskIDs, err := orchestrator.persistSubtasks(event, subtaskDrafts, blastContext)

	if err != nil {
		orchestrator.moveCard(event.ID, "backlog", "subtask persist error: "+err.Error())
		return
	}

	// Fan out: each subtask gets its own goroutine bounded by the semaphore.
	// All subtasks share the same feature branch and the same file lock registry.
	var subtaskWG sync.WaitGroup
	errCh := make(chan error, len(subtaskIDs))

	allSubtasks, err := orchestrator.subtasks.ListForCard(orchestrator.ctx, event.ID)

	if err != nil {
		orchestrator.moveCard(event.ID, "backlog", "subtask list error: "+err.Error())
		return
	}

	for _, subtask := range allSubtasks {
		orchestrator.sem <- struct{}{}
		subtaskWG.Add(1)

		go func(st Subtask) {
			defer func() {
				<-orchestrator.sem
				subtaskWG.Done()
			}()

			if err := orchestrator.runSubtask(event, st, branch); err != nil {
				errCh <- fmt.Errorf("subtask %q: %w", st.Title, err)
			}
		}(subtask)
	}

	subtaskWG.Wait()
	close(errCh)

	var errs []string

	for err := range errCh {
		errs = append(errs, err.Error())
	}

	if len(errs) > 0 {
		orchestrator.moveCard(event.ID, "backlog", "subtask failures: "+strings.Join(errs, "; "))
		return
	}

	// All subtasks done — commit, push, open PR.
	// We use a final scratch sandbox just for the commit/push since each
	// subtask sandbox was destroyed after its own work.
	if err := orchestrator.finalise(event, branch); err != nil {
		orchestrator.moveCard(event.ID, "backlog", "finalise error: "+err.Error())
		return
	}
}

/*
runPlanner spins up a short-lived sandbox for the Planner agent (read-only
access to the repo), runs the Planner, and returns the drafted subtasks.
*/
func (orchestrator *Orchestrator) runPlanner(
	event ColumnEvent, blastContext string,
) ([]SubtaskDraft, error) {
	agentID := "planner-" + event.ID[:8]

	sandbox, err := NewSandbox(orchestrator.ctx, SandboxConfig{
		AgentID:       agentID,
		Image:         orchestrator.cfg.DockerImage,
		GitHubToken:   orchestrator.cfg.GitHubToken,
		GitHubOwner:   orchestrator.cfg.GitHubOwner,
		GitHubRepo:    orchestrator.cfg.GitHubRepo,
		FeatureBranch: featureBranch(event),
	})

	if err != nil {
		return nil, fmt.Errorf("planner sandbox: %w", err)
	}

	defer sandbox.Destroy()

	if err := sandbox.Start(); err != nil {
		return nil, fmt.Errorf("planner sandbox start: %w", err)
	}

	editor := NewVirtualEditor(agentID, sandbox, orchestrator.locks)
	planner := NewPlanner(orchestrator.ctx, orchestrator.cfg.Planner, editor)

	result, err := planner.Plan(event.Title, event.Description, blastContext)

	if err != nil {
		return nil, err
	}

	return result.Subtasks, nil
}

/*
persistSubtasks writes the Planner's drafts to the database, annotating each
subtask's context snapshot with sibling awareness so developers know what
neighbouring agents are working on.
*/
func (orchestrator *Orchestrator) persistSubtasks(
	event ColumnEvent,
	drafts []SubtaskDraft,
	blastContext string,
) ([]string, error) {
	ids := make([]string, 0, len(drafts))

	for i, draft := range drafts {
		snap := SubtaskContext{
			BlastRadius:  blastContext,
			KeySymbols:   draft.KeySymbols,
			FilesInScope: draft.FilesInScope,
			SiblingNotes: draft.SiblingNotes,
		}

		id, err := orchestrator.subtasks.Insert(
			orchestrator.ctx, event.ID, i, draft.Title, draft.Description, snap,
		)

		if err != nil {
			return nil, err
		}

		ids = append(ids, id)
	}

	return ids, nil
}

/*
runSubtask executes the full developer→reviewer loop for a single subtask in
its own ephemeral sandbox on the shared feature branch.
*/
func (orchestrator *Orchestrator) runSubtask(
	event ColumnEvent, subtask Subtask, branch string,
) error {
	agentID := "dev-" + subtask.ID[:8]

	defer orchestrator.locks.ReleaseAll(agentID)

	_ = orchestrator.subtasks.SetStatus(orchestrator.ctx, subtask.ID, "in-progress", agentID)

	sandbox, err := NewSandbox(orchestrator.ctx, SandboxConfig{
		AgentID:       agentID,
		Image:         orchestrator.cfg.DockerImage,
		GitHubToken:   orchestrator.cfg.GitHubToken,
		GitHubOwner:   orchestrator.cfg.GitHubOwner,
		GitHubRepo:    orchestrator.cfg.GitHubRepo,
		FeatureBranch: branch,
	})

	if err != nil {
		_ = orchestrator.subtasks.SetStatus(orchestrator.ctx, subtask.ID, "failed", agentID)
		return fmt.Errorf("sandbox: %w", err)
	}

	defer sandbox.Destroy()

	if err := sandbox.Start(); err != nil {
		_ = orchestrator.subtasks.SetStatus(orchestrator.ctx, subtask.ID, "failed", agentID)
		return fmt.Errorf("sandbox start: %w", err)
	}

	editor := NewVirtualEditor(agentID, sandbox, orchestrator.locks)
	developer := NewDeveloper(orchestrator.ctx, orchestrator.cfg.Developer, editor)
	reviewer := NewReviewer(orchestrator.ctx, orchestrator.cfg.Reviewer)

	// Build the subtask-scoped context block injected into the developer prompt.
	subtaskContext := formatSubtaskContext(subtask)

	feedback := ""

	for range maxIterations {
		if err := developer.Implement(
			subtask.Title, subtask.Description, subtaskContext, feedback,
		); err != nil {
			_ = orchestrator.subtasks.SetStatus(orchestrator.ctx, subtask.ID, "failed", agentID)
			return err
		}

		verdict, err := reviewer.Review(sandbox, subtask.Title, subtask.Description)

		if err != nil {
			_ = orchestrator.subtasks.SetStatus(orchestrator.ctx, subtask.ID, "failed", agentID)
			return err
		}

		if verdict.Pass {
			break
		}

		feedback = verdict.Feedback
	}

	commitMsg := fmt.Sprintf("feat(%s): %s", event.ID[:8], subtask.Title)

	if err := sandbox.CommitAndPush(commitMsg); err != nil {
		_ = orchestrator.subtasks.SetStatus(orchestrator.ctx, subtask.ID, "failed", agentID)
		return fmt.Errorf("commit/push: %w", err)
	}

	_ = orchestrator.subtasks.SetStatus(orchestrator.ctx, subtask.ID, "done", agentID)

	return nil
}

/*
finalise opens the pull request for the completed feature branch and moves the
parent card to the review column.
*/
func (orchestrator *Orchestrator) finalise(event ColumnEvent, branch string) error {
	prURL, err := orchestrator.openPR(event, branch)

	if err != nil {
		return err
	}

	orchestrator.moveCard(event.ID, "review", prURL)

	return nil
}

func (orchestrator *Orchestrator) moveCard(cardID, columnKey, note string) {
	q := `UPDATE kanban_cards SET column_key = $1, updated_at = NOW() WHERE id = $2`
	_, _ = orchestrator.db.ExecContext(orchestrator.ctx, q, columnKey, cardID)

	if note == "" {
		return
	}

	desc := fmt.Sprintf("[devteam] %s", note)
	_, _ = orchestrator.db.ExecContext(
		orchestrator.ctx,
		`UPDATE kanban_cards SET description = description || E'\n' || $1 WHERE id = $2`,
		desc, cardID,
	)
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

func (orchestrator *Orchestrator) extractContext(event ColumnEvent) string {
	keywords := keywordsFromCard(event.Title, event.Description)
	radius, err := orchestrator.extractor.Extract(".", keywords, orchestrator.cfg.BlastRadiusDepth)

	if err != nil || radius == nil {
		return ""
	}

	return radius.Format()
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

func keywordsFromCard(title, description string) []string {
	combined := title + " " + description
	words := strings.FieldsFunc(combined, func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= 'A' && r <= 'Z') && !(r >= '0' && r <= '9')
	})

	seen := make(map[string]struct{})
	keywords := make([]string, 0, len(words))

	for _, word := range words {
		lower := strings.ToLower(word)

		if len(lower) < 3 {
			continue
		}

		if _, ok := seen[lower]; ok {
			continue
		}

		seen[lower] = struct{}{}
		keywords = append(keywords, lower)
	}

	return keywords
}

/*
formatSubtaskContext renders the subtask's stored context snapshot into a
markdown block for injection into the developer agent's system prompt.
*/
func formatSubtaskContext(subtask Subtask) string {
	snap := subtask.ContextSnapshot
	var sb strings.Builder

	if snap.BlastRadius != "" {
		sb.WriteString(snap.BlastRadius)
		sb.WriteString("\n")
	}

	if len(snap.FilesInScope) > 0 {
		sb.WriteString("### Files in scope for this subtask\n")

		for _, f := range snap.FilesInScope {
			fmt.Fprintf(&sb, "- %s\n", f)
		}

		sb.WriteString("\n")
	}

	if len(snap.KeySymbols) > 0 {
		sb.WriteString("### Key symbols\n")

		for _, sym := range snap.KeySymbols {
			fmt.Fprintf(&sb, "- `%s`\n", sym)
		}

		sb.WriteString("\n")
	}

	if len(snap.SiblingNotes) > 0 {
		sb.WriteString("### Sibling subtask conflicts to be aware of\n")

		for title, note := range snap.SiblingNotes {
			fmt.Fprintf(&sb, "- **%s**: %s\n", title, note)
		}

		sb.WriteString("\n")
	}

	return sb.String()
}
