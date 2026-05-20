package compute

import (
	"context"
	"errors"
	"sync"

	"github.com/theapemachine/caramba/pkg/errnie/validate"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu"
	"github.com/theapemachine/puter/device/metal"
	"github.com/theapemachine/qpool"
)

var ErrDeviceNotFound = errors.New("compute: device not found")
var ErrExecutorUnavailable = errors.New("compute: executor unavailable for device")

/*
Backend routes manifest and program execution to discovered devices.
It owns every active Location at once (host SIMD and Metal GPU together).
Placement policy and parallel scheduling belong in a separate optimizer
type composed on top of Backend later.
*/
type Backend struct {
	ctx       context.Context
	cancel    context.CancelFunc
	devices   map[DeviceID]device.Backend
	mesh      tensor.ShardingMesh
	pool      *qpool.Q
	closeOnce sync.Once
}

func NewBackend(ctx context.Context, pool *qpool.Q) (*Backend, error) {
	ctx, cancel := context.WithCancel(ctx)

	cpuBackend, err := cpu.NewBackend(ctx, pool)
	if err != nil {
		cancel()
		return nil, err
	}

	metalBackend, err := metal.NewBackend(ctx, pool)

	if err != nil {
		cancel()
		return nil, err
	}

	backend := &Backend{
		ctx:    ctx,
		cancel: cancel,
		devices: map[DeviceID]device.Backend{
			DeviceID{Location: tensor.Host, Index: 0}:  cpuBackend,
			DeviceID{Location: tensor.Metal, Index: 0}: metalBackend,
		},
		mesh: tensor.ShardingMesh{
			Devices:   []tensor.Location{tensor.Metal},
			Shape:     []int{1},
			AxisNames: []string{"device"},
		},
		pool: pool,
	}

	return backend, validate.Require(map[string]any{
		"ctx":     ctx,
		"devices": backend.devices,
		"mesh":    backend.mesh,
	})
}

func (backend *Backend) Device(id DeviceID) (device.Backend, error) {
	device, ok := backend.devices[id]
	if !ok {
		return nil, ErrDeviceNotFound
	}
	return device, nil
}

func (backend *Backend) Close() error {
	return nil
}
