package qpool

import (
	"context"
	"fmt"
	"runtime/debug"
	"time"

	"github.com/phuslu/log"
	"golang.org/x/sync/errgroup"
)

func processJob(q *Q, workerCtx context.Context, job Job) {
	if err := waitDependencies(q, workerCtx, job); err != nil {
		latency := time.Since(job.StartTime)
		q.metrics.RecordJobOutcome(latency, false)

		if job.CircuitID != "" {
			if cb := q.breakerFor(&job); cb != nil {
				cb.RecordFailure()
			}
		}

		q.publishTelemetry(Event{
			Component: "qpool",
			Op:        "job-error",
			Message:   fmt.Sprintf("dependencies unmet: %s (%v)", job.ID, err),
			Time:      time.Now(),
			Level:     log.WarnLevel,
			Err:       err,
			Fields: []Field{
				{Key: "job", Value: job.ID},
				{Key: "phase", Value: "dependency"},
				{Key: "duration_ms", Value: latency.Milliseconds()},
			},
		})

		q.space.StoreError(job.ID, err, job.TTL)

		return
	}

	deadline := q.schedulingTimeout()

	if job.ExecTimeout > 0 {
		deadline = job.ExecTimeout
	}

	execCtx, cancel := context.WithTimeout(workerCtx, deadline)
	defer cancel()

	startedAt := time.Now()

	q.publishTelemetry(Event{
		Component: "qpool",
		Op:        "job-start",
		Message:   fmt.Sprintf("job started: %s", job.ID),
		Time:      startedAt,
		Level:     log.InfoLevel,
		Fields: []Field{
			{Key: "job", Value: job.ID},
		},
	})

	result, err := runJobWithRetries(execCtx, job)

	latency := time.Since(job.StartTime)
	execDur := time.Since(startedAt)

	if err != nil {
		q.metrics.RecordJobOutcome(latency, false)

		if job.CircuitID != "" {
			if cb := q.breakerFor(&job); cb != nil {
				cb.RecordFailure()
			}
		}

		q.publishTelemetry(Event{
			Component: "qpool",
			Op:        "job-error",
			Message:   fmt.Sprintf("job failed: %s (%v)", job.ID, err),
			Time:      time.Now(),
			Level:     log.ErrorLevel,
			Err:       err,
			Fields: []Field{
				{Key: "job", Value: job.ID},
				{Key: "phase", Value: "execution"},
				{Key: "duration_ms", Value: latency.Milliseconds()},
				{Key: "exec_duration_ms", Value: execDur.Milliseconds()},
			},
		})

		q.space.StoreError(job.ID, err, job.TTL)

		return
	}

	q.metrics.RecordJobOutcome(latency, true)

	if job.CircuitID != "" {
		if cb := q.breakerFor(&job); cb != nil {
			cb.RecordSuccess()
		}
	}

	q.publishTelemetry(Event{
		Component: "qpool",
		Op:        "job-complete",
		Message:   fmt.Sprintf("job completed: %s in %s", job.ID, latency.Round(time.Millisecond)),
		Time:      time.Now(),
		Level:     log.InfoLevel,
		Fields: []Field{
			{Key: "job", Value: job.ID},
			{Key: "duration_ms", Value: latency.Milliseconds()},
			{Key: "exec_duration_ms", Value: execDur.Milliseconds()},
		},
	})

	q.space.Store(job.ID, result, job.TTL)
}

func waitDependencies(q *Q, workerCtx context.Context, job Job) error {
	if len(job.Dependencies) == 0 {
		return nil
	}

	eg, egCtx := errgroup.WithContext(workerCtx)

	for _, depID := range job.Dependencies {
		eg.Go(func() error {
			return waitOneDependency(q, egCtx, job, depID)
		})
	}

	return eg.Wait()
}

func dependencyAwaitTimeout(policy *RetryPolicy, strategy RetryStrategy) time.Duration {
	if policy != nil && policy.PerAttemptTimeout > 0 {
		return policy.PerAttemptTimeout
	}

	var base time.Duration

	if strategy != nil {
		base = strategy.NextDelay(1)
	}

	if base <= 0 {
		base = time.Second
	}

	const maxDerived = 60 * time.Second

	if base > maxDerived {
		return maxDerived
	}

	if base < time.Second {
		return time.Second
	}

	return base
}

func waitOneDependency(q *Q, workerCtx context.Context, job Job, depID string) error {
	maxAttempts := 1

	strategy := RetryStrategy(&ExponentialBackoff{Initial: time.Second})
	var lastErr error

	if job.DependencyRetryPolicy != nil {
		maxAttempts = job.DependencyRetryPolicy.MaxAttempts

		if job.DependencyRetryPolicy.Strategy != nil {
			strategy = job.DependencyRetryPolicy.Strategy
		}
	}

	awaitTimeout := dependencyAwaitTimeout(job.DependencyRetryPolicy, strategy)

	for attempt := 0; attempt < maxAttempts; attempt++ {
		ch := q.space.Await(depID)

		waitCtx, cancel := context.WithTimeout(workerCtx, awaitTimeout)

		select {
		case result := <-ch:
			cancel()

			if result == nil {
				lastErr = fmt.Errorf("dependency %s returned nil result", depID)

				if attempt < maxAttempts-1 {
					time.Sleep(strategy.NextDelay(attempt + 1))
				}

				continue
			}

			if result.Error != nil {
				return fmt.Errorf("dependency %s: %w", depID, result.Error)
			}

			return nil

		case <-waitCtx.Done():
			lastErr = waitCtx.Err()
			cancel()

			if err := workerCtx.Err(); err != nil {
				return fmt.Errorf("dependency %s: %w", depID, err)
			}

			if attempt < maxAttempts-1 {
				time.Sleep(strategy.NextDelay(attempt + 1))
			}
		}
	}

	q.space.RegisterDependent(depID, job.ID)

	if lastErr != nil {
		return fmt.Errorf("dependency %s failed after %d attempts: %w", depID, maxAttempts, lastErr)
	}

	return fmt.Errorf("dependency %s failed after %d attempts", depID, maxAttempts)
}

func runJobWithRetries(ctx context.Context, job Job) (any, error) {
	maxAttempts := 1

	strategy := RetryStrategy(&ExponentialBackoff{Initial: time.Second})

	if job.RetryPolicy != nil {
		if job.RetryPolicy.MaxAttempts > 0 {
			maxAttempts = job.RetryPolicy.MaxAttempts
		}

		if job.RetryPolicy.Strategy != nil {
			strategy = job.RetryPolicy.Strategy
		}
	}

	var lastErr error

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		res, err := invokeFnOnce(ctx, job)

		if err == nil {
			return res, nil
		}

		lastErr = err

		if job.RetryPolicy != nil && job.RetryPolicy.Filter != nil && !job.RetryPolicy.Filter(err) {
			break
		}

		if attempt == maxAttempts {
			break
		}

		delay := strategy.NextDelay(attempt)

		if delay <= 0 {
			delay = time.Millisecond
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
	}

	return nil, lastErr
}

func invokeFnOnce(ctx context.Context, job Job) (res any, err error) {
	defer func() {
		if r := recover(); r != nil {
			res = nil

			err = fmt.Errorf("qpool: panic in job %s: %v\n%s", job.ID, r, debug.Stack())
		}
	}()

	return job.Fn(ctx)
}
