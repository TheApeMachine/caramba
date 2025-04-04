package twoface

import (
	"context"

	"github.com/theapemachine/caramba/pkg/datura"
)

/*
Job is an interface any type can implement if they want to be able to use the
generics goroutine pool.
*/
type Job interface {
	Do(*datura.Artifact, chan *datura.Artifact) *datura.Artifact
}

/*
NewJob is a conveniance method to convert any incoming structured type to a
Job interface such that they can get onto the worker pools.
*/
func NewJob(jobType Job) Job {
	return jobType
}

/*
RetriableJob provides boilerplate for quickly building jobs that
retry based on a backoff delay strategy.
*/
type RetriableJob struct {
	ctx    context.Context
	cancel context.CancelFunc
	fn     Job
	tries  int
	out    chan *datura.Artifact
}

func NewRetriableJob(ctx context.Context, fn Job, tries int, out chan *datura.Artifact) Job {
	ctx, cancel := context.WithCancel(ctx)

	return NewJob(RetriableJob{
		ctx:    ctx,
		cancel: cancel,
		fn:     fn,
		tries:  tries,
		out:    out,
	})
}

/*
Do the job and retry x amount of times when needed.
*/
func (job RetriableJob) Do(artifact *datura.Artifact, out chan *datura.Artifact) *datura.Artifact {
	return NewRetrier(NewFibonacci(job.tries)).Do(job.fn, artifact, out)
}
