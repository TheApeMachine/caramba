package config

import (
	"context"
	"runtime"
	"time"

	"github.com/theapemachine/qpool"
)

var qpoolRootKey = "qpool"

/*
QPoolConfig holds worker pool settings loaded from config.yml.
*/
type QPoolConfig struct {
	MinWorkers         int
	MaxWorkers         int
	SchedulingTimeout  time.Duration
	JobChannelCapacity int
}

/*
NewQPoolConfig reads qpool settings from the loaded config.yml.
*/
func NewQPoolConfig() *QPoolConfig {
	defaultWorkers := runtime.NumCPU()

	if defaultWorkers < 2 {
		defaultWorkers = 2
	}

	maxWorkers := defaultWorkers * 2

	return &QPoolConfig{
		MinWorkers: WithDefault(qpoolRootKey+".min_workers", defaultWorkers),
		MaxWorkers: WithDefault(qpoolRootKey+".max_workers", maxWorkers),
		SchedulingTimeout: WithDefault(
			qpoolRootKey+".scheduling_timeout",
			10*time.Second,
		),
		JobChannelCapacity: WithDefault(
			qpoolRootKey+".job_channel_capacity",
			0,
		),
	}
}

/*
NewWorkerPool constructs the process-wide qpool from config.
*/
func (qpoolConfig *QPoolConfig) NewWorkerPool(ctx context.Context) *qpool.Q {
	poolConfig := qpool.NewConfig()
	poolConfig.SchedulingTimeout = qpoolConfig.SchedulingTimeout

	if qpoolConfig.JobChannelCapacity > 0 {
		poolConfig.JobChannelCapacity = qpoolConfig.JobChannelCapacity
	}

	poolConfig.TelemetryPublish = qpool.Publish

	return qpool.NewQ(
		ctx,
		qpoolConfig.MinWorkers,
		qpoolConfig.MaxWorkers,
		poolConfig,
	)
}
