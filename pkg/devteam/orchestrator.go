package devteam

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/go-github/v67/github"
	_ "github.com/lib/pq"
	"golang.org/x/oauth2"

	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/caramba/pkg/qpool"
)

const devteamJobTimeout = 24 * time.Hour

const reviewerMaxIterations = 10

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
	ctx           context.Context
	cancel        context.CancelFunc
	cfg           *config.DevTeamConfig
	db            *sql.DB
	subtasks      *SubtaskStore
	github        *github.Client
	watcher       eventWatcher
	locks         *FileLockRegistry
	extractor     *ContextExtractor
	watcherPool   *qpool.Q
	cardPool      *qpool.Q
	subtaskPool   *qpool.Q
	integrationMu sync.Mutex
	activeMu      sync.Mutex
	activeCards   map[string]struct{}
}

type eventWatcher interface {
	Events() <-chan ColumnEvent
	Errors() <-chan error
	Watch() error
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
		ctx:         ctx,
		cancel:      cancel,
		cfg:         cfg,
		db:          db,
		subtasks:    NewSubtaskStore(db),
		github:      ghClient,
		watcher:     NewWatcher(ctx, cfg.DatabaseURL),
		locks:       NewFileLockRegistry(ctx),
		extractor:   extractor,
		watcherPool: newDevTeamPool(ctx, "watcher", 1),
		cardPool:    newDevTeamPool(ctx, "card", maxConcurrent(cfg.MaxConcurrent)),
		subtaskPool: newDevTeamPool(ctx, "subtask", maxConcurrent(cfg.MaxConcurrent)),
		activeCards: map[string]struct{}{},
	}, nil
}

/*
Run starts the watcher through qpool and processes column-change events until
ctx is cancelled or Stop is called.
*/
func (orchestrator *Orchestrator) Run() error {
	orchestrator.watcherPool.Schedule(
		"devteam.watcher",
		func(context.Context) (any, error) {
			if err := orchestrator.watcher.Watch(); err != nil {
				orchestrator.cancel()

				return nil, err
			}

			return nil, nil
		},
		qpool.WithExecTimeout(devteamJobTimeout),
	)

	events := orchestrator.watcher.Events()
	errors := orchestrator.watcher.Errors()

	for {
		select {
		case <-orchestrator.ctx.Done():
			return nil

		case err, ok := <-errors:
			if !ok {
				return nil
			}

			if err != nil {
				qpool.Publish(qpool.NewWarningEvent(
					"devteam",
					"watcher",
					"watcher reconnecting after error",
					[]qpool.Field{{Key: "error", Value: err.Error()}},
				))

				continue
			}

		case event, ok := <-events:
			if !ok {
				orchestrator.cancel()

				return fmt.Errorf("orchestrator: watcher events closed")
			}

			if !orchestrator.isRelevant(event) {
				continue
			}

			cardEvent := event

			if !orchestrator.claimCard(cardEvent.ID) {
				continue
			}

			orchestrator.cardPool.Schedule(
				"devteam.card."+cardEvent.ID,
				func(context.Context) (any, error) {
					defer orchestrator.releaseCard(cardEvent.ID)
					orchestrator.handle(cardEvent)

					return nil, nil
				},
				qpool.WithExecTimeout(devteamJobTimeout),
			)
		}
	}
}

func (orchestrator *Orchestrator) claimCard(cardID string) bool {
	orchestrator.activeMu.Lock()
	defer orchestrator.activeMu.Unlock()

	if orchestrator.activeCards == nil {
		orchestrator.activeCards = map[string]struct{}{}
	}

	if _, exists := orchestrator.activeCards[cardID]; exists {
		return false
	}

	orchestrator.activeCards[cardID] = struct{}{}

	return true
}

func (orchestrator *Orchestrator) releaseCard(cardID string) {
	orchestrator.activeMu.Lock()
	defer orchestrator.activeMu.Unlock()

	delete(orchestrator.activeCards, cardID)
}

/*
Stop cancels the orchestrator and waits for all in-flight qpool jobs to finish.
*/
func (orchestrator *Orchestrator) Stop() {
	orchestrator.cancel()
	orchestrator.closePools()
	orchestrator.locks.Close()
	orchestrator.extractor.Close()
	_ = orchestrator.db.Close()
}

func (orchestrator *Orchestrator) closePools() {
	if orchestrator.watcherPool != nil {
		orchestrator.watcherPool.Close()
		orchestrator.watcherPool = nil
	}

	if orchestrator.cardPool != nil {
		orchestrator.cardPool.Close()
		orchestrator.cardPool = nil
	}

	if orchestrator.subtaskPool != nil {
		orchestrator.subtaskPool.Close()
		orchestrator.subtaskPool = nil
	}
}

func (orchestrator *Orchestrator) isRelevant(event ColumnEvent) bool {
	return event.ColumnKey == "todo" &&
		event.ResearchProjectID == orchestrator.cfg.RequestsProjectID
}

func maxConcurrent(value int) int {
	if value > 0 {
		return value
	}

	return 1
}

func newDevTeamPool(ctx context.Context, name string, workers int) *qpool.Q {
	workers = maxConcurrent(workers)

	return qpool.NewQ(
		ctx,
		workers,
		workers,
		&qpool.Config{
			SchedulingTimeout:  devteamJobTimeout,
			JobChannelCapacity: workers * 4,
			Scaler:             nil,
			TelemetryPublish: func(event qpool.Event) {
				event.WithField("devteam_pool", name)
				qpool.Publish(event)
			},
		},
	)
}

/*
handle is the top-level card lifecycle:
 1. Extract blast radius from the host repo.
 2. Spin up a planner sandbox, run the Planner agent, persist subtasks.
 3. Fan out: each subtask gets its own sandbox and developer→reviewer loop.
 4. Wait for all subtasks; move card to review (or backlog on failure).
*/
func (orchestrator *Orchestrator) handle(event ColumnEvent) {
	if err := orchestrator.moveCard(event.ID, "in-progress", ""); err != nil {
		return
	}

	blastContext := orchestrator.extractContext(event)

	subtaskDrafts, err := orchestrator.runPlanner(event, blastContext)

	if err != nil {
		_ = orchestrator.moveCard(event.ID, "backlog", "planner error: "+err.Error())
		return
	}

	if len(subtaskDrafts) == 0 {
		_ = orchestrator.moveCard(event.ID, "backlog", "planner produced no subtasks")
		return
	}

	branch := featureBranch(event)

	subtaskIDs, err := orchestrator.persistSubtasks(event, subtaskDrafts, blastContext)

	if err != nil {
		_ = orchestrator.moveCard(event.ID, "backlog", "subtask persist error: "+err.Error())
		return
	}

	allSubtasks, err := orchestrator.subtasks.ListForCard(orchestrator.ctx, event.ID)

	if err != nil {
		_ = orchestrator.moveCard(event.ID, "backlog", "subtask list error: "+err.Error())
		return
	}

	results := make([]chan *qpool.QValue, 0, len(subtaskIDs))

	for _, subtask := range allSubtasks {
		subtask := subtask

		results = append(results, orchestrator.subtaskPool.Schedule(
			"devteam.subtask."+subtask.ID,
			func(context.Context) (any, error) {
				if err := orchestrator.runSubtask(event, subtask, branch); err != nil {
					return nil, fmt.Errorf("subtask %q: %w", subtask.Title, err)
				}

				return nil, nil
			},
			qpool.WithExecTimeout(devteamJobTimeout),
		))
	}

	var errs []string

	for _, result := range results {
		select {
		case <-orchestrator.ctx.Done():
			errs = append(errs, orchestrator.ctx.Err().Error())
		case value, channelOpen := <-result:
			if !channelOpen {
				errs = append(errs, "subtask result channel closed")
				continue
			}

			if value != nil && value.Error != nil {
				errs = append(errs, value.Error.Error())
			}
		}
	}

	if len(errs) > 0 {
		_ = orchestrator.moveCard(event.ID, "backlog", "subtask failures: "+strings.Join(errs, "; "))
		return
	}

	// All subtasks done — commit, push, open PR.
	// We use a final scratch sandbox just for the commit/push since each
	// subtask sandbox was destroyed after its own work.
	if err := orchestrator.finalise(event, branch); err != nil {
		_ = orchestrator.moveCard(event.ID, "backlog", "finalise error: "+err.Error())
		return
	}
}
