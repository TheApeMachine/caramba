package twoface

import (
	"bytes"
	"context"
	"time"

	"github.com/theapemachine/caramba/pkg/datura"
)

/*
Worker wraps a concurrent process that is able to process Job types
scheduled onto a Pool.
*/
type Worker struct {
	ctx     context.Context
	cancel  context.CancelFunc
	pool    *Pool
	jobs    chan Job
	buffer  *bytes.Buffer
	latency time.Duration
}

func NewWorker(pool *Pool) *Worker {
	ctx, cancel := context.WithCancel(context.Background())

	return &Worker{
		ctx:     ctx,
		cancel:  cancel,
		pool:    pool,
		jobs:    make(chan Job),
		buffer:  bytes.NewBuffer([]byte{}),
		latency: 0 * time.Nanosecond,
	}
}

func (worker *Worker) Generate(
	artifact chan *datura.Artifact,
) chan *datura.Artifact {
	go func() {
		var data *datura.Artifact

		for {
			select {
			case <-worker.pool.ctx.Done():
				worker.Close()
			case <-worker.ctx.Done():
				return
			case data = <-artifact:
				worker.pool.workers <- worker.jobs
				job := <-worker.jobs

				t := time.Now()
				job.Do(data)

				worker.latency = time.Duration(
					time.Since(t).Nanoseconds(),
				)
			}
		}

	}()

	return artifact
}

func (worker *Worker) Close() error {
	worker.cancel()
	return nil
}
