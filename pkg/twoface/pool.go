package twoface

import (
	"context"
	"sync"

	"github.com/theapemachine/caramba/pkg/datura"
)

var (
	oncePool sync.Once
	pool     *Pool
)

/*
Pool is a set of Worker types, each running their own (pre-warmed) goroutine.
Any object that implements the Job interface is able to schedule work on the
worker pool, which keeps the amount of goroutines in check, while still being
able to benefit from high concurrency in all kinds of scenarios.
*/
type Pool struct {
	ctx     context.Context
	workers chan chan Job
	jobs    chan Job
}

/*
NewPool instantiates a worker pool with bound size of maxWorkers, taking in a
Context type to be able to cleanly cancel all of the sub processes it starts.
*/
func NewPool(ctx context.Context) *Pool {
	oncePool.Do(func() {
		pool = &Pool{
			ctx:     ctx,
			workers: make(chan chan Job),
			jobs:    make(chan Job, 1),
		}
	})

	return pool
}

/*
Submit runs a function concurrently in the pool.
*/
func (pool *Pool) Submit(fn func()) {
	pool.Do(NewJob(funcJob(fn)))
}

// funcJob wraps a function to implement the Job interface
type funcJob func()

func (f funcJob) Do(artifact *datura.Artifact) *datura.Artifact {
	f()
	return artifact
}

/*
Do is the entry point for new jobs that want to be scheduled onto the worker pool.
*/
func (pool *Pool) Do(jobType Job) {
	// Send the job to the job channel.
	pool.jobs <- NewJob(jobType)
}

/*
Run the workers, after creating and assigning them to the pool.
*/
func (pool *Pool) Run() *Pool {
	// Start the auto-scaler to control the pool size dynamically.
	NewScaler(pool)

	// Start the job scheduling process.
	go pool.dispatch()
	return pool
}

func (pool *Pool) dispatch() {
	// Make sure that we cleanly close the channels if our dispatcher
	// returns for whatever reason.
	defer close(pool.jobs)
	defer close(pool.workers)

	for {
		select {
		case job := <-pool.jobs:
			// A new job was received from the jobs queue, get the first available
			// worker from the pool once ready.
			jobChannel := <-pool.workers
			// Then send the job to the worker for processing.
			jobChannel <- job
		case <-pool.ctx.Done():
			return
		}
	}
}
