package qpool

import (
	"context"
	"fmt"
	"maps"
	"sync"
	"sync/atomic"
	"time"

	"github.com/phuslu/log"
)

type breakerMap struct {
	entries map[string]*CircuitBreaker
}

type workerToken struct {
	id     uint64
	cancel context.CancelFunc
}

type workerRegistry struct {
	tokens []*workerToken
}

/*
Q combines a buffered job queue, fixed worker set, optional regulators, and result tracking via QSpace.
*/
type Q struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	jobCh        chan Job
	shutdownMu   sync.RWMutex
	stopping     atomic.Bool
	closeJobOnce sync.Once
	minWorkers   int
	maxWorkers   int

	space   *QSpace
	scaler  *Scaler
	metrics *Metrics

	breakers   atomic.Pointer[breakerMap]
	registry   atomic.Pointer[workerRegistry]
	nextWorker atomic.Uint64

	config *Config
}

/*
NewQ constructs a pool with minWorkers..maxWorkers goroutines competing on a shared job channel.
*/
func NewQ(ctx context.Context, minWorkers, maxWorkers int, config *Config) *Q {
	if config == nil {
		config = NewConfig()
	}

	if maxWorkers < 1 {
		maxWorkers = 1
	}

	if minWorkers < 1 {
		minWorkers = 1
	}

	if minWorkers > maxWorkers {
		minWorkers = maxWorkers
	}

	ctx, cancel := context.WithCancel(ctx)

	capacity := maxWorkers * 10

	if config.JobChannelCapacity > 0 {
		capacity = config.JobChannelCapacity
	}

	q := &Q{
		ctx:        ctx,
		cancel:     cancel,
		jobCh:      make(chan Job, capacity),
		minWorkers: minWorkers,
		maxWorkers: maxWorkers,
		space:      NewQSpace(),
		metrics:    NewMetrics(),
		config:     config,
	}

	for i := 0; i < minWorkers; i++ {
		q.startWorker()
	}

	if config.Scaler != nil {
		q.scaler = NewScaler(q, minWorkers, maxWorkers, config.Scaler)
	}

	return q
}

/*
MetricSnapshot returns a point-in-time copy of atomic pool counters (workers,
busy workers, queue depth, and regulator-facing fields).
*/
func (q *Q) MetricSnapshot() MetricReading {
	if q == nil {
		return MetricReading{}
	}

	r := q.metrics.CollectReading()

	return *r
}

/*
WorkerBounds returns the configured minimum and maximum worker goroutine counts.
*/
func (q *Q) WorkerBounds() (minWorkers, maxWorkers int) {
	if q == nil {
		return 0, 0
	}

	return q.minWorkers, q.maxWorkers
}

/*
PeriodicScalerConfigured reports whether NewQ wired the built-in interval scaler.

Adaptive admission regulators may resize the pool independently; those are not mirrored here.
*/
func (q *Q) PeriodicScalerConfigured() bool {
	return q != nil && q.config != nil && q.config.Scaler != nil
}

func (q *Q) publishTelemetry(ev Event) {
	if q != nil && q.config != nil && q.config.TelemetryPublish != nil {
		q.config.TelemetryPublish(ev)

		return
	}

	Publish(ev)
}

func (q *Q) schedulingTimeout() time.Duration {
	if q.config != nil && q.config.SchedulingTimeout > 0 {
		return q.config.SchedulingTimeout
	}

	return 5 * time.Second
}

/*
Schedule enqueues a job when regulators and optional circuit breaker permit.
Results arrive on the returned channel backed by QSpace. The job id doubles as
the result key until TTL expires — reuse the same id for a logically new piece
of work while older results remain queued and callers will unblock with the
stale completion first unless result cleanup removed it first.
*/
func (q *Q) Schedule(id string, fn func(context.Context) (any, error), opts ...JobOption) chan *QValue {
	ctx, cancel := context.WithTimeout(q.ctx, q.schedulingTimeout())
	defer cancel()

	startTime := time.Now()

	job := Job{
		ID:        id,
		Fn:        fn,
		StartTime: startTime,
		RetryPolicy: &RetryPolicy{
			MaxAttempts: 1,
			Strategy:    &ExponentialBackoff{Initial: time.Second},
		},
	}

	for _, opt := range opts {
		opt(&job)
	}

	if q.config != nil && len(q.config.Regulators) > 0 {
		reading := q.metrics.CollectReading()

		for _, reg := range q.config.Regulators {
			reg.Observe(reading)
		}

		for _, reg := range q.config.Regulators {
			if reg.Limit() {
				q.metrics.incThrottled()

				ch := make(chan *QValue, 1)
				ch <- &QValue{
					Error:     fmt.Errorf("qpool: regulator rejected schedule"),
					CreatedAt: time.Now(),
				}

				close(ch)

				return ch
			}
		}
	}

	if job.CircuitID != "" {
		breaker := q.breakerFor(&job)

		if breaker != nil && !breaker.Allow() {
			ch := make(chan *QValue, 1)
			ch <- &QValue{
				Error:     fmt.Errorf("circuit breaker %s is open", job.CircuitID),
				CreatedAt: time.Now(),
			}

			close(ch)

			return ch
		}
	}

	if q.stopping.Load() {
		ch := make(chan *QValue, 1)
		ch <- &QValue{
			Error:     fmt.Errorf("qpool: pool closed"),
			CreatedAt: time.Now(),
		}

		close(ch)

		return ch
	}

	q.shutdownMu.RLock()

	if q.stopping.Load() {
		q.shutdownMu.RUnlock()

		ch := make(chan *QValue, 1)
		ch <- &QValue{
			Error:     fmt.Errorf("qpool: pool closed"),
			CreatedAt: time.Now(),
		}

		close(ch)

		return ch
	}

	if q.ctx.Err() != nil {
		q.shutdownMu.RUnlock()

		ch := make(chan *QValue, 1)
		ch <- &QValue{
			Error:     fmt.Errorf("qpool: pool closed: %w", q.ctx.Err()),
			CreatedAt: time.Now(),
		}

		close(ch)

		return ch
	}

	select {
	case <-q.ctx.Done():
		q.shutdownMu.RUnlock()

		ch := make(chan *QValue, 1)
		ch <- &QValue{
			Error:     fmt.Errorf("qpool: pool closed: %w", q.ctx.Err()),
			CreatedAt: time.Now(),
		}

		close(ch)

		return ch

	case q.jobCh <- job:
		q.publishTelemetry(Event{
			Component: "qpool",
			Op:        "schedule",
			Message:   fmt.Sprintf("job scheduled: %s", id),
			Time:      time.Now(),
			Level:     log.InfoLevel,
		})

		q.metrics.incJobQueued()
		q.shutdownMu.RUnlock()

		return q.space.Await(id)

	case <-ctx.Done():
		q.shutdownMu.RUnlock()

		ch := make(chan *QValue, 1)
		ch <- &QValue{
			Error:     fmt.Errorf("job scheduling timeout: %w", ctx.Err()),
			CreatedAt: time.Now(),
		}

		close(ch)
		q.metrics.incSchedulingFailure()

		return ch
	}
}

/*
CreateBroadcastGroup allocates a group stored inside QSpace.
*/
func (q *Q) CreateBroadcastGroup(id string, ttl time.Duration) *BroadcastGroup {
	return q.space.CreateBroadcastGroup(id, ttl)
}

/*
Subscribe returns the broadcast group's subscriber channel for groupID.
*/
func (q *Q) Subscribe(groupID string) chan *QValue {
	return q.space.Subscribe(groupID)
}

/*
PeekResult returns a shallow copy of the stored QValue for job id when QSpace holds a non-expired result.

It returns (nil, false) when no result is stored for id, when TTL expiration or eviction removed the entry, or when the pool or space cannot serve the query (including during shutdown).

The returned *QValue points at a new struct value copied from the actor's map entry; see QSpace.PeekResult for concurrency and read-only semantics versus nested reference fields in QValue.
*/
func (q *Q) PeekResult(id string) (*QValue, bool) {
	if q == nil {
		return nil, false
	}

	return q.space.PeekResult(id)
}

/*
WithTTL sets how long QSpace retains the job result before expiration
cleanup. It does not cap execution time; use WithExecTimeout for that.
*/
func WithTTL(ttl time.Duration) JobOption {
	return func(j *Job) {
		j.TTL = ttl
	}
}

/*
WithExecTimeout sets the per-invocation deadline passed to Fn. Zero selects
the pool Config.SchedulingTimeout default (when positive) or five seconds.
*/
func WithExecTimeout(d time.Duration) JobOption {
	return func(j *Job) {
		j.ExecTimeout = d
	}
}

/*
WithDependencyAwaitTimeout sets how long a job waits for each dependency before
its dependency wait attempt times out. It does not add dependencies; combine it
with WithDependencies for dependency-ordered jobs.
*/
func WithDependencyAwaitTimeout(d time.Duration) JobOption {
	return func(j *Job) {
		if d <= 0 {
			return
		}

		if j.DependencyRetryPolicy == nil {
			j.DependencyRetryPolicy = &RetryPolicy{
				MaxAttempts: 1,
				Strategy:    &ExponentialBackoff{Initial: time.Second},
			}
		}

		if j.DependencyRetryPolicy.MaxAttempts <= 0 {
			j.DependencyRetryPolicy.MaxAttempts = 1
		}

		if j.DependencyRetryPolicy.Strategy == nil {
			j.DependencyRetryPolicy.Strategy = &ExponentialBackoff{Initial: time.Second}
		}

		j.DependencyRetryPolicy.PerAttemptTimeout = d
	}
}

func (q *Q) breakerFor(job *Job) *CircuitBreaker {
	if job.CircuitID == "" || job.CircuitConfig == nil {
		return nil
	}

	for {
		cur := q.breakers.Load()

		var m map[string]*CircuitBreaker

		if cur != nil {
			m = cur.entries

			if b, ok := m[job.CircuitID]; ok {
				return b
			}
		}

		newM := make(map[string]*CircuitBreaker)

		maps.Copy(newM, m)

		cb := newCircuitBreakerFromConfig(job.CircuitConfig)
		newM[job.CircuitID] = cb
		next := &breakerMap{entries: newM}

		if q.breakers.CompareAndSwap(cur, next) {
			return cb
		}
	}
}

func (q *Q) registryPush(tok *workerToken) {
	for {
		old := q.registry.Load()

		var cur []*workerToken

		if old != nil {
			cur = old.tokens
		}

		nextSlice := append(append([]*workerToken{}, cur...), tok)
		next := &workerRegistry{tokens: nextSlice}

		if q.registry.CompareAndSwap(old, next) {
			return
		}
	}
}

func (q *Q) registryPopLast() *workerToken {
	for {
		old := q.registry.Load()

		if old == nil || len(old.tokens) == 0 {
			return nil
		}

		cur := old.tokens
		last := cur[len(cur)-1]
		nextSlice := make([]*workerToken, len(cur)-1)

		copy(nextSlice, cur[:len(cur)-1])
		next := &workerRegistry{tokens: nextSlice}

		if q.registry.CompareAndSwap(old, next) {
			return last
		}
	}
}

func (q *Q) registryRemove(id uint64) {
	for {
		old := q.registry.Load()

		if old == nil {
			return
		}

		cur := old.tokens
		idx := -1

		for i, t := range cur {
			if t.id == id {
				idx = i
				break
			}
		}

		if idx < 0 {
			return
		}

		nextSlice := append(append([]*workerToken{}, cur[:idx]...), cur[idx+1:]...)
		next := &workerRegistry{tokens: nextSlice}

		if q.registry.CompareAndSwap(old, next) {
			return
		}
	}
}

func (q *Q) startWorker() {
	if !q.metrics.tryIncWorkerIfBelow(q.maxWorkers) {
		return
	}

	workerCtx, cancel := context.WithCancel(q.ctx)

	id := q.nextWorker.Add(1)
	tok := &workerToken{id: id, cancel: cancel}

	q.registryPush(tok)

	q.wg.Go(func() {
		q.runWorker(workerCtx, tok)
	})

	q.publishTelemetry(Event{
		Component: "qpool",
		Op:        "worker-start",
		Message:   fmt.Sprintf("worker started; workers=%d", q.metrics.workerCount.Load()),
		Time:      time.Now(),
		Level:     log.DebugLevel,
		Fields: []Field{
			{Key: "workers", Value: q.metrics.workerCount.Load()},
		},
	})
}

func (q *Q) runWorker(workerCtx context.Context, tok *workerToken) {
	defer q.registryRemove(tok.id)
	defer q.metrics.decWorkerCount()

	for {
		select {
		case <-workerCtx.Done():
			q.publishTelemetry(Event{
				Component: "qpool",
				Op:        "worker-exit",
				Message:   "worker exiting due to cancellation",
				Time:      time.Now(),
				Level:     log.DebugLevel,
				Fields: []Field{
					{Key: "worker", Value: tok.id},
				},
			})

			return

		case job, ok := <-q.jobCh:
			if !ok {
				return
			}

			q.metrics.decJobQueued()
			q.metrics.incBusyWorker()

			func() {
				defer q.metrics.decBusyWorker()

				processJob(q, workerCtx, job)
			}()
		}
	}
}

func (q *Q) scaleDownWorkers(count int) {
	for range count {
		tok := q.registryPopLast()

		if tok == nil {
			return
		}

		tok.cancel()
	}
}

/*
Close cancels workers and drains queued jobs with shutdown errors.
*/
func (q *Q) Close() {
	if q == nil {
		return
	}

	q.publishTelemetry(Event{
		Component: "qpool",
		Op:        "close",
		Message:   "closing Q pool",
		Time:      time.Now(),
		Level:     log.DebugLevel,
	})

	q.shutdownMu.Lock()
	q.stopping.Store(true)

	if q.cancel != nil {
		q.cancel()
	}

	q.shutdownMu.Unlock()
	q.wg.Wait()
	q.shutdownMu.Lock()

	q.closeJobOnce.Do(func() {
		close(q.jobCh)
	})

	for job := range q.jobCh {
		q.space.StoreError(job.ID, fmt.Errorf("qpool: pool shut down"), job.TTL)
	}

	q.shutdownMu.Unlock()
	q.space.Close()
	q.publishTelemetry(Event{
		Component: "qpool",
		Op:        "closed",
		Message:   "Q pool closed",
		Time:      time.Now(),
		Level:     log.DebugLevel,
	})
}
