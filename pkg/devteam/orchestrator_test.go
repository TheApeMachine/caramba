package devteam

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"io"
	"sync/atomic"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/caramba/pkg/qpool"
)

func init() {
	sql.Register("devteam_orchestrator_test", &orchestratorTestDriver{})
}

func TestOrchestratorRun(t *testing.T) {
	Convey("Given an Orchestrator with a failing watcher", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		watcher := newOrchestratorTestWatcher(errors.New("watch failed"))
		orchestrator := newTestOrchestrator(ctx, cancel, watcher)
		defer orchestrator.closePools()

		Convey("It should return the watcher error instead of blocking", func() {
			err := orchestrator.Run()

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "watch failed")
		})
	})

	Convey("Given an Orchestrator with a closed events channel", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		watcher := newOrchestratorTestWatcher(nil)
		watcher.closeEvents()

		orchestrator := newTestOrchestrator(ctx, cancel, watcher)
		defer orchestrator.closePools()

		Convey("It should return instead of waiting forever", func() {
			err := orchestrator.Run()

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "events closed")
		})
	})
}

func TestFeatureBranch(t *testing.T) {
	Convey("Given a ColumnEvent with a feature title", t, func() {
		event := ColumnEvent{
			ID:    "f47ac10b-58cc-4372-a567-0e02b2c3d479",
			Title: "Add dark mode support",
		}

		Convey("It should produce a slug-safe branch name", func() {
			branch := featureBranch(event)

			So(branch, ShouldStartWith, "devteam/")
			So(branch, ShouldContainSubstring, "f47ac10b")
			So(branch, ShouldContainSubstring, "add")
			So(branch, ShouldContainSubstring, "dark")
			So(branch, ShouldNotContainSubstring, " ")
		})
	})

	Convey("Given a very long feature title", t, func() {
		event := ColumnEvent{
			ID:    "aaaaaaaa-0000-0000-0000-000000000000",
			Title: "implement a very long feature that has many words in the title beyond limit",
		}

		Convey("It should truncate to at most 6 slug words", func() {
			branch := featureBranch(event)
			parts := splitBranchSlug(branch)
			// At most 6 words after the ID segment.
			So(len(parts), ShouldBeLessThanOrEqualTo, 6)
		})
	})
}

func TestIsRelevant(t *testing.T) {
	Convey("Given an Orchestrator configured for the requests project", t, func() {
		cfg := &config.DevTeamConfig{
			RequestsProjectID: "f47ac10b-58cc-4372-a567-0e02b2c3d479",
		}

		orchestrator := &Orchestrator{
			ctx: context.Background(),
			cfg: cfg,
		}

		Convey("It should accept todo events on the requests project", func() {
			event := ColumnEvent{
				ColumnKey:         "todo",
				ResearchProjectID: "f47ac10b-58cc-4372-a567-0e02b2c3d479",
			}

			So(orchestrator.isRelevant(event), ShouldBeTrue)
		})

		Convey("It should reject events on a different column", func() {
			event := ColumnEvent{
				ColumnKey:         "in-progress",
				ResearchProjectID: "f47ac10b-58cc-4372-a567-0e02b2c3d479",
			}

			So(orchestrator.isRelevant(event), ShouldBeFalse)
		})

		Convey("It should reject events on a different project", func() {
			event := ColumnEvent{
				ColumnKey:         "todo",
				ResearchProjectID: "00000000-0000-0000-0000-000000000001",
			}

			So(orchestrator.isRelevant(event), ShouldBeFalse)
		})
	})
}

func TestOrchestrator_moveCard(t *testing.T) {
	Convey("Given an orchestrator backed by a database", t, func() {
		database, err := sql.Open("devteam_orchestrator_test", "")
		So(err, ShouldBeNil)
		defer func() { So(database.Close(), ShouldBeNil) }()

		orchestrator := &Orchestrator{
			ctx: context.Background(),
			db:  database,
		}

		orchestratorTestExecCount.Store(0)
		orchestratorTestExecFailure.Store(false)

		Convey("It should update the card column and note", func() {
			err := orchestrator.moveCard(
				"f47ac10b-58cc-4372-a567-0e02b2c3d479",
				"review",
				"ready",
			)

			So(err, ShouldBeNil)
			So(orchestratorTestExecCount.Load(), ShouldEqual, 2)
		})

		Convey("It should return SQL errors instead of swallowing them", func() {
			orchestratorTestExecFailure.Store(true)

			err := orchestrator.moveCard(
				"f47ac10b-58cc-4372-a567-0e02b2c3d479",
				"review",
				"ready",
			)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "move card")
			So(err.Error(), ShouldContainSubstring, "test exec failure")
			So(orchestratorTestExecCount.Load(), ShouldEqual, 1)
		})
	})
}

func TestOrchestratorConcurrency(t *testing.T) {
	Convey("Given separate card and subtask pools", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		orchestrator := &Orchestrator{
			ctx:         ctx,
			cardPool:    newDevTeamPool(ctx, "card-test", 1),
			subtaskPool: newDevTeamPool(ctx, "subtask-test", 1),
		}
		defer orchestrator.closePools()

		Convey("It should run a subtask job while a card job is held", func() {
			releaseCard := make(chan struct{})
			cardResult := orchestrator.cardPool.Schedule(
				"card-block",
				func(context.Context) (any, error) {
					<-releaseCard

					return nil, nil
				},
				qpool.WithExecTimeout(time.Second),
			)
			subtaskResult := orchestrator.subtaskPool.Schedule(
				"subtask-independent",
				func(context.Context) (any, error) {
					return "done", nil
				},
				qpool.WithExecTimeout(time.Second),
			)

			select {
			case result := <-subtaskResult:
				So(result, ShouldNotBeNil)
				So(result.Error, ShouldBeNil)
				So(result.Value, ShouldEqual, "done")
			case <-time.After(250 * time.Millisecond):
				t.Fatal("subtask pool deadlocked behind card pool")
			}

			close(releaseCard)

			select {
			case result := <-cardResult:
				So(result, ShouldNotBeNil)
				So(result.Error, ShouldBeNil)
			case <-time.After(time.Second):
				t.Fatal("card pool did not release")
			}
		})
	})

	Convey("Given invalid concurrency", t, func() {
		Convey("It should clamp to one slot", func() {
			So(maxConcurrent(0), ShouldEqual, 1)
			So(maxConcurrent(-4), ShouldEqual, 1)
			So(maxConcurrent(3), ShouldEqual, 3)
		})
	})
}

func TestOrchestratorIntegrationLock(t *testing.T) {
	Convey("Given concurrent branch integration attempts", t, func() {
		orchestrator := &Orchestrator{}
		entered := make(chan int, 2)
		release := make(chan struct{})
		done := make(chan struct{}, 2)

		integrate := func(id int) {
			orchestrator.integrationMu.Lock()
			defer orchestrator.integrationMu.Unlock()

			entered <- id
			<-release
			done <- struct{}{}
		}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		pool := newDevTeamPool(ctx, "integration-test", 2)
		defer pool.Close()

		firstResult := pool.Schedule(
			"integrate-1",
			func(context.Context) (any, error) {
				integrate(1)

				return nil, nil
			},
			qpool.WithExecTimeout(time.Second),
		)
		So(<-entered, ShouldEqual, 1)

		secondResult := pool.Schedule(
			"integrate-2",
			func(context.Context) (any, error) {
				integrate(2)

				return nil, nil
			},
			qpool.WithExecTimeout(time.Second),
		)

		select {
		case <-entered:
			t.Fatal("second integration entered before first released")
		case <-time.After(100 * time.Millisecond):
		}

		close(release)
		<-done
		So(<-entered, ShouldEqual, 2)
		<-done

		So((<-firstResult).Error, ShouldBeNil)
		So((<-secondResult).Error, ShouldBeNil)
	})
}

func BenchmarkFeatureBranch(b *testing.B) {
	event := ColumnEvent{
		ID:    "f47ac10b-58cc-4372-a567-0e02b2c3d479",
		Title: "Add real-time collaboration to the kanban board",
	}

	for b.Loop() {
		_ = featureBranch(event)
	}
}

func BenchmarkOrchestrator_moveCard(b *testing.B) {
	database, err := sql.Open("devteam_orchestrator_test", "")

	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = database.Close() }()

	orchestrator := &Orchestrator{
		ctx: context.Background(),
		db:  database,
	}
	orchestratorTestExecFailure.Store(false)
	b.ResetTimer()

	for b.Loop() {
		if err := orchestrator.moveCard("card", "review", "ready"); err != nil {
			b.Fatal(err)
		}
	}
}

// splitBranchSlug splits the word segments after the ID prefix.
func splitBranchSlug(branch string) []string {
	// branch format: devteam/<8-char-id>-word1-word2-...
	// skip "devteam/" prefix (8 chars) + "/" + 8-char id + "-"
	if len(branch) < len("devteam/")+9 {
		return nil
	}

	rest := branch[len("devteam/")+9:]
	words := make([]string, 0)
	current := ""

	for _, ch := range rest {
		if ch == '-' {
			if current != "" {
				words = append(words, current)
				current = ""
			}
		} else {
			current += string(ch)
		}
	}

	if current != "" {
		words = append(words, current)
	}

	return words
}

var orchestratorTestExecCount atomic.Int64
var orchestratorTestExecFailure atomic.Bool

type orchestratorTestDriver struct{}

func (driver *orchestratorTestDriver) Open(name string) (driver.Conn, error) {
	return &orchestratorTestConn{}, nil
}

type orchestratorTestConn struct{}

func (connection *orchestratorTestConn) Prepare(query string) (driver.Stmt, error) {
	return nil, errors.New("prepare is not supported")
}

func (connection *orchestratorTestConn) Close() error {
	return nil
}

func (connection *orchestratorTestConn) Begin() (driver.Tx, error) {
	return nil, errors.New("transactions are not supported")
}

func (connection *orchestratorTestConn) ExecContext(
	ctx context.Context,
	query string,
	args []driver.NamedValue,
) (driver.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	orchestratorTestExecCount.Add(1)

	if orchestratorTestExecFailure.Load() {
		return nil, errors.New("test exec failure")
	}

	return driver.RowsAffected(1), nil
}

func (connection *orchestratorTestConn) QueryContext(
	ctx context.Context,
	query string,
	args []driver.NamedValue,
) (driver.Rows, error) {
	return orchestratorTestRows{}, nil
}

type orchestratorTestRows struct{}

func (rows orchestratorTestRows) Columns() []string {
	return nil
}

func (rows orchestratorTestRows) Close() error {
	return nil
}

func (rows orchestratorTestRows) Next(dest []driver.Value) error {
	return io.EOF
}

type orchestratorTestWatcher struct {
	events chan ColumnEvent
	errors chan error
	done   chan struct{}
	err    error
}

func newOrchestratorTestWatcher(err error) *orchestratorTestWatcher {
	return &orchestratorTestWatcher{
		events: make(chan ColumnEvent),
		errors: make(chan error, 1),
		done:   make(chan struct{}),
		err:    err,
	}
}

func newTestOrchestrator(
	ctx context.Context,
	cancel context.CancelFunc,
	watcher *orchestratorTestWatcher,
) *Orchestrator {
	return &Orchestrator{
		ctx:         ctx,
		cancel:      cancel,
		watcher:     watcher,
		watcherPool: newDevTeamPool(ctx, "watcher-test", 1),
		cardPool:    newDevTeamPool(ctx, "card-test", 1),
		subtaskPool: newDevTeamPool(ctx, "subtask-test", 1),
	}
}

func (watcher *orchestratorTestWatcher) Events() <-chan ColumnEvent {
	return watcher.events
}

func (watcher *orchestratorTestWatcher) Errors() <-chan error {
	return watcher.errors
}

func (watcher *orchestratorTestWatcher) Watch() error {
	if watcher.err == nil {
		<-watcher.done

		return nil
	}

	watcher.errors <- watcher.err
	time.Sleep(time.Millisecond)

	return nil
}

func (watcher *orchestratorTestWatcher) closeEvents() {
	close(watcher.events)
	close(watcher.done)
}
