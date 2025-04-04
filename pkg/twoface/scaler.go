package twoface

import (
	"time"
)

/*
Scaler is a process that evaluates the resource load of a Pool and
adds or removes Workers according to its opinion about how to divide
the machine resources available.
*/
type Scaler struct {
	pool    *Pool
	workers []*Worker
	current *state
}

type state struct {
	prev  int64
	ok    bool
	trend bool
}

func NewScaler(pool *Pool) {
	scaler := &Scaler{pool, make([]*Worker, 0), &state{0, true, false}}
	scaler.initialize()
}

func (scaler *Scaler) initialize() {
	go func() {
		var latency int64

		for {
			scaler.current.prev = latency
			latency = 0

			// Get the total latency of all active workers, as a representative
			// value of the "load" on the underlying system.
			for _, worker := range scaler.workers {
				latency += worker.latency.Nanoseconds()
			}

			// If we are currently scaled down to zero, we need to get
			// some workers going first, before we can start scaling them.
			if latency == 0 && len(scaler.workers) == 0 {
				for i := 0; i < 1; i++ {
					worker := NewWorker(scaler.pool)
					scaler.workers = append(scaler.workers, worker)
					worker.Generate(nil)
				}

				continue
			}

			// Sleep for 100 milliseconds so we don't overload the scaling.
			time.Sleep(100 * time.Millisecond)

			// First step is to identify the differency between the
			// previous latency, to see if we are up or down.
			if latency > scaler.current.prev && scaler.current.ok {
				// Latency going up.
				scaler.current.ok = false
				scaler.current.trend = false
				continue
			}

			if latency < scaler.current.prev && !scaler.current.ok {
				// Latency going down.
				scaler.current.ok = true
				scaler.current.trend = false
				continue
			}

			if !scaler.current.trend {
				// We are observing a trend.
				scaler.current.trend = true
				continue
			}

			switch scaler.current.ok {
			case true:
				// Scale the worker pool up.
				scaler.workers = append(
					scaler.workers, NewWorker(scaler.pool),
				)
			case false:
				// Scale the worker pool down.
				var worker *Worker
				worker, scaler.workers = scaler.workers[0], scaler.workers[1:]
				worker.Close()
			}

			scaler.current = &state{latency, true, false}
		}
	}()
}
