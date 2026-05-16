package devteam

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/theapemachine/caramba/pkg/qpool"
)

/*
ClaimResult is returned by FileLockRegistry.Claim. When Acquired is false the
holder's AgentID and stated Intent are provided so the requesting agent can
decide whether to wait or adjust its approach.
*/
type ClaimResult struct {
	Acquired bool
	HolderID string
	Intent   string
}

type fileClaim struct {
	agentID string
	intent  string
	at      time.Time
}

type lockOp func(map[string]*fileClaim)

/*
FileLockRegistry is a shared, process-wide registry that lets concurrent agent
jobs declare intent to modify a file before touching it. It is backed by a
single qpool actor job so all state mutations are serialised without explicit
locking.

Each Docker sandbox runs in its own container (separate filesystem) but the
source repository is pushed via CommitAndPush to a shared remote branch
namespace. Claiming a path here prevents two agents from generating conflicting
changes to the same file; the second agent learns about the first agent's intent
and can adapt its implementation accordingly.
*/
type FileLockRegistry struct {
	ctx      context.Context
	cancel   context.CancelFunc
	ops      chan lockOp
	done     chan struct{}
	doneOnce sync.Once
	stopped  atomic.Bool
	pool     *qpool.Q
	result   chan *qpool.QValue
}

/*
NewFileLockRegistry starts the actor loop. Call Close when the Orchestrator
shuts down.
*/
func NewFileLockRegistry(ctx context.Context) *FileLockRegistry {
	ctx, cancel := context.WithCancel(ctx)

	registry := &FileLockRegistry{
		ctx:    ctx,
		cancel: cancel,
		ops:    make(chan lockOp, 1024),
		done:   make(chan struct{}),
	}

	registry.pool = qpool.NewQ(
		ctx,
		1,
		1,
		&qpool.Config{
			SchedulingTimeout:  24 * time.Hour,
			JobChannelCapacity: 2,
			Scaler:             nil,
		},
	)
	registry.result = registry.pool.Schedule(
		"devteam.filelock.loop",
		func(jobCtx context.Context) (any, error) {
			registry.loop(jobCtx)

			return nil, nil
		},
		qpool.WithExecTimeout(24*time.Hour),
	)

	return registry
}

func (registry *FileLockRegistry) loop(ctx context.Context) {
	defer registry.closeDone()

	claims := make(map[string]*fileClaim)

	for {
		select {
		case op := <-registry.ops:
			op(claims)
		case <-ctx.Done():
			return
		case <-registry.ctx.Done():
			return
		}
	}
}

func (registry *FileLockRegistry) closeDone() {
	registry.doneOnce.Do(func() {
		close(registry.done)
	})
}

func (registry *FileLockRegistry) submit(op lockOp) bool {
	if registry.stopped.Load() {
		return false
	}

	select {
	case registry.ops <- op:
		return true
	case <-registry.done:
		return false
	}
}

/*
Claim attempts to acquire an exclusive lock on path for agentID. Intent is a
short human-readable description of what the agent plans to do (e.g. "add
error handling to Foo"). If another agent already holds the claim the result
has Acquired=false and carries that agent's ID and intent.
*/
func (registry *FileLockRegistry) Claim(agentID, path, intent string) ClaimResult {
	reply := make(chan ClaimResult, 1)

	registry.submit(func(claims map[string]*fileClaim) {
		existing, held := claims[path]

		if held && existing.agentID != agentID {
			reply <- ClaimResult{
				Acquired: false,
				HolderID: existing.agentID,
				Intent:   existing.intent,
			}

			return
		}

		claims[path] = &fileClaim{
			agentID: agentID,
			intent:  intent,
			at:      time.Now(),
		}

		reply <- ClaimResult{Acquired: true}
	})

	select {
	case result := <-reply:
		return result
	case <-registry.done:
		return ClaimResult{Acquired: false, HolderID: "registry-closed"}
	}
}

/*
Release removes the claim for path held by agentID. A no-op if the path is
not claimed or is claimed by a different agent.
*/
func (registry *FileLockRegistry) Release(agentID, path string) {
	registry.submit(func(claims map[string]*fileClaim) {
		if existing, ok := claims[path]; ok && existing.agentID == agentID {
			delete(claims, path)
		}
	})
}

/*
ReleaseAll removes every claim held by agentID. Called when a card lifecycle
ends (success or failure) to unblock any waiting agents.
*/
func (registry *FileLockRegistry) ReleaseAll(agentID string) {
	registry.submit(func(claims map[string]*fileClaim) {
		for path, claim := range claims {
			if claim.agentID == agentID {
				delete(claims, path)
			}
		}
	})
}

/*
Snapshot returns a copy of all current claims. Useful for injecting awareness
into an agent's context: "these files are being modified by other agents right now".
*/
func (registry *FileLockRegistry) Snapshot() map[string]string {
	reply := make(chan map[string]string, 1)

	registry.submit(func(claims map[string]*fileClaim) {
		snapshot := make(map[string]string, len(claims))

		for path, claim := range claims {
			snapshot[path] = fmt.Sprintf("%s: %s", claim.agentID, claim.intent)
		}

		reply <- snapshot
	})

	select {
	case snap := <-reply:
		return snap
	case <-registry.done:
		return nil
	}
}

/*
Close stops the actor loop. All subsequent Claim calls return Acquired=false.
*/
func (registry *FileLockRegistry) Close() {
	if registry.stopped.Swap(true) {
		return
	}

	registry.cancel()

	if registry.pool != nil {
		registry.pool.Close()
		registry.pool = nil
	}

	registry.closeDone()
}
