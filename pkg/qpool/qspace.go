package qpool

import (
	"fmt"
	"sync/atomic"
	"time"
)

/*
QSpace stores job results, waits, broadcast groups, and dependency edges behind a single actor goroutine.
*/
type QSpace struct {
	ops      chan func(*spaceState)
	shutdown chan struct{}
	done     chan struct{}
	stopped  atomic.Bool
}

type spaceState struct {
	values          map[string]*QValue
	waiting         map[string][]chan *QValue
	groups          map[string]*BroadcastGroup
	children        map[string]map[string]struct{}
	parents         map[string]map[string]struct{}
	cleanupInterval time.Duration
}

/*
NewQSpace starts the actor loop with periodic expiration passes.
*/
func NewQSpace() *QSpace {
	qs := &QSpace{
		ops:      make(chan func(*spaceState), 4096),
		shutdown: make(chan struct{}),
		done:     make(chan struct{}),
	}

	go qs.loop()

	return qs
}

/*
submit enqueues fn on the actor or returns false once the space is torn
down so callers do not block indefinitely after Close.
*/
func (qs *QSpace) submit(fn func(*spaceState)) bool {
	if qs.stopped.Load() {
		return false
	}

	select {
	case <-qs.done:
		return false
	case qs.ops <- fn:
		return true
	}
}

func (qs *QSpace) loop() {
	defer close(qs.done)

	st := &spaceState{
		values:          make(map[string]*QValue),
		waiting:         make(map[string][]chan *QValue),
		groups:          make(map[string]*BroadcastGroup),
		children:        make(map[string]map[string]struct{}),
		parents:         make(map[string]map[string]struct{}),
		cleanupInterval: time.Minute,
	}

	ticker := time.NewTicker(st.cleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case fn := <-qs.ops:
			if fn != nil {
				fn(st)
			}

		case <-ticker.C:
			qspaceCleanup(st)

		case <-qs.shutdown:
			for {
				select {
				case fn := <-qs.ops:
					if fn != nil {
						fn(st)
					}
				default:
					qspaceShutdown(st)

					return
				}
			}
		}
	}
}

/*
Store persists a completed value and fulfills waiters.
*/
func (qs *QSpace) Store(id string, value interface{}, ttl time.Duration) {
	if !qs.submit(func(st *spaceState) {
		qv := NewQValue(value)
		qv.TTL = ttl
		st.values[id] = qv

		if channels, ok := st.waiting[id]; ok {
			for _, ch := range append([]chan *QValue(nil), channels...) {
				select {
				case ch <- qv:
				default:
				}
			}

			delete(st.waiting, id)
		}
	}) {
		return
	}
}

/*
Await returns a channel that receives exactly one *QValue for id.
*/
func (qs *QSpace) Await(id string) chan *QValue {
	replySync := make(chan chan *QValue, 1)

	closedRecv := func() chan *QValue {
		ch := make(chan *QValue, 1)

		close(ch)

		return ch
	}

	work := func(st *spaceState) {
		out := make(chan *QValue, 1)

		if qv, ok := st.values[id]; ok {
			out <- qv
			close(out)
			replySync <- out

			return
		}

		st.waiting[id] = append(st.waiting[id], out)
		replySync <- out
	}

	if !qs.submit(work) {
		return closedRecv()
	}

	select {
	case <-qs.done:
		return closedRecv()
	case out := <-replySync:
		return out
	}
}

/*
PeekResult returns (nil, false) when id has no stored completion, when the entry was removed by TTL-based cleanup in the actor loop (see qspaceCleanup and QValue.TTL), or when the space is stopped and the operation cannot complete.

When ok is true, it returns a shallow copy of the map's QValue: the QValue struct itself is copied onto a new heap pointer, which is race-free with respect to mutations to the internal map entry concurrent with Store — but fields such as QValue.Value (interface{}) and QValue.Error may still reference shared underlying objects. Callers must treat the returned QValue as read-only, or deep-copy before mutating those fields. Mutating nested state concurrently with Store or with other readers is unsafe.

Concurrent calls to PeekResult alongside Store are safe for map lookup and for receiving this copied QValue shell; they do not serialize access to reference-type data inside QValue. Dependent jobs typically call PeekResult inside the job Fn after WithDependencies has synchronized execution with producers that already called Store on the dependency ids.
*/
func (qs *QSpace) PeekResult(id string) (*QValue, bool) {
	type peekReply struct {
		qv *QValue
		ok bool
	}

	reply := make(chan peekReply, 1)

	if !qs.submit(func(st *spaceState) {
		v, ok := st.values[id]

		if !ok {
			reply <- peekReply{nil, false}

			return
		}

		cp := *v

		reply <- peekReply{&cp, true}
	}) {
		return nil, false
	}

	select {
	case <-qs.done:
		return nil, false
	case r := <-reply:
		return r.qv, r.ok
	}
}

/*
Exists reports whether id currently has a stored value.
*/
func (qs *QSpace) Exists(id string) bool {
	if qs.stopped.Load() {
		return false
	}

	reply := make(chan bool, 1)

	if !qs.submit(func(st *spaceState) {
		_, ok := st.values[id]
		reply <- ok
	}) {
		return false
	}

	select {
	case <-qs.done:
		return false
	case v := <-reply:
		return v
	}
}

/*
StoreError stores a terminal error result for id.
*/
func (qs *QSpace) StoreError(id string, err error, ttl time.Duration) {
	if !qs.submit(func(st *spaceState) {
		qv := NewQValue(nil)
		qv.Error = err
		qv.TTL = ttl
		st.values[id] = qv

		if channels, ok := st.waiting[id]; ok {
			for _, ch := range append([]chan *QValue(nil), channels...) {
				select {
				case ch <- qv:
				default:
				}
			}

			delete(st.waiting, id)
		}
	}) {
		return
	}
}

/*
AddRelationship records a dependency edge parent -> child.
*/
func (qs *QSpace) AddRelationship(parentID, childID string) error {
	if qs.stopped.Load() {
		return fmt.Errorf("qpool: space closed")
	}

	reply := make(chan error, 1)

	if !qs.submit(func(st *spaceState) {
		if qspaceWouldCreateCircle(st, parentID, childID) {
			reply <- fmt.Errorf("qpool: circular dependency detected")

			return
		}

		qspaceAddEdge(st, parentID, childID)
		reply <- nil
	}) {
		return fmt.Errorf("qpool: space closed")
	}

	select {
	case <-qs.done:
		return fmt.Errorf("qpool: space closed")
	case err := <-reply:
		return err
	}
}

/*
RegisterDependent records that jobID waits on depID when dependency polling fails.

This mirrors prior bookkeeping used for relational jobs without exposing mutexes.
*/
func (qs *QSpace) RegisterDependent(depID, jobID string) {
	_ = qs.submit(func(st *spaceState) {
		qspaceAddEdge(st, depID, jobID)
	})
}

/*
CreateBroadcastGroup registers a pub/sub group owned by this space.
*/
func (qs *QSpace) CreateBroadcastGroup(id string, ttl time.Duration) *BroadcastGroup {
	if qs.stopped.Load() {
		bg := NewBroadcastGroup(id, ttl, 1)

		bg.Close()

		return bg
	}

	reply := make(chan *BroadcastGroup, 1)

	if !qs.submit(func(st *spaceState) {
		if qs.stopped.Load() {
			bg := NewBroadcastGroup(id, ttl, 1)

			bg.Close()

			reply <- bg

			return
		}

		group := NewBroadcastGroup(id, ttl, 100)
		st.groups[id] = group
		reply <- group
	}) {
		bg := NewBroadcastGroup(id, ttl, 1)

		bg.Close()

		return bg
	}

	select {
	case <-qs.done:
		bg := NewBroadcastGroup(id, ttl, 1)

		bg.Close()

		return bg
	case g := <-reply:
		return g
	}
}

/*
Subscribe attaches to a broadcast group by id.
*/
func (qs *QSpace) Subscribe(groupID string) chan *QValue {
	if qs.stopped.Load() {
		ch := make(chan *QValue)

		close(ch)

		return ch
	}

	reply := make(chan chan *QValue, 1)

	if !qs.submit(func(st *spaceState) {
		if qs.stopped.Load() {
			ch := make(chan *QValue)

			close(ch)

			reply <- ch

			return
		}

		if group, ok := st.groups[groupID]; ok {
			reply <- group.Subscribe("", 10)

			return
		}

		dummy := make(chan *QValue)

		close(dummy)

		reply <- dummy
	}) {
		ch := make(chan *QValue)

		close(ch)

		return ch
	}

	select {
	case <-qs.done:
		ch := make(chan *QValue)

		close(ch)

		return ch
	case ch := <-reply:
		return ch
	}
}

/*
Close stops maintenance and releases channels.
*/
func (qs *QSpace) Close() {
	if qs.stopped.Swap(true) {
		return
	}

	close(qs.shutdown)
	<-qs.done
}

func qspaceAddEdge(st *spaceState, parentID, childID string) {
	if st.children[parentID] == nil {
		st.children[parentID] = make(map[string]struct{})
	}

	st.children[parentID][childID] = struct{}{}

	if st.parents[childID] == nil {
		st.parents[childID] = make(map[string]struct{})
	}

	st.parents[childID][parentID] = struct{}{}
}

func qspaceCleanup(st *spaceState) {
	now := time.Now()

	for id, qv := range st.values {
		if qv.TTL <= 0 {
			continue
		}

		if now.Sub(qv.CreatedAt) <= qv.TTL {
			continue
		}

		delete(st.values, id)
		qspacePruneEdges(st, id)
	}

	for id, group := range st.groups {
		if group == nil {
			continue
		}

		if !group.Expired(now) {
			continue
		}

		group.Close()
		delete(st.groups, id)
	}
}

func qspacePruneEdges(st *spaceState, id string) {
	delete(st.children, id)

	for parentID, childSet := range st.children {
		delete(childSet, id)

		if len(childSet) == 0 {
			delete(st.children, parentID)
		}
	}

	delete(st.parents, id)

	for childID, parentSet := range st.parents {
		delete(parentSet, id)

		if len(parentSet) == 0 {
			delete(st.parents, childID)
		}
	}
}

func qspaceShutdown(st *spaceState) {
	for _, channels := range st.waiting {
		for _, ch := range channels {
			close(ch)
		}
	}

	for _, group := range st.groups {
		if group != nil {
			group.Close()
		}
	}

	st.waiting = nil
	st.values = nil
	st.groups = nil
	st.children = nil
	st.parents = nil
}

func qspaceWouldCreateCircle(st *spaceState, parentID, childID string) bool {
	visited := make(map[string]bool)

	var dfs func(string) bool

	dfs = func(current string) bool {
		if current == parentID {
			return true
		}

		if visited[current] {
			return false
		}

		visited[current] = true

		for p := range st.parents[current] {
			if dfs(p) {
				return true
			}
		}

		return false
	}

	return dfs(childID)
}
