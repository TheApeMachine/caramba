package compute

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/theapemachine/caramba/pkg/errnie/validate"
	"github.com/theapemachine/caramba/pkg/qpool"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
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
	devices   []*Device
	byID      map[DeviceID]*Device
	mesh      tensor.ShardingMesh
	pool      *qpool.Q
	closeOnce sync.Once
}

func NewBackend(ctx context.Context, pool *qpool.Q) (*Backend, error) {
	ctx, cancel := context.WithCancel(ctx)

	devices, err := discoverDevices()

	if err != nil {
		cancel()

		return nil, err
	}

	backend := &Backend{
		ctx:     ctx,
		cancel:  cancel,
		devices: devices,
		byID:    indexDevices(devices),
		mesh:    buildMesh(devices),
		pool:    pool,
	}

	return backend, validate.Require(map[string]any{
		"ctx":     ctx,
		"devices": backend.devices,
		"mesh":    backend.mesh,
	})
}

func (backend *Backend) Context() context.Context {
	if backend == nil {
		return context.Background()
	}

	return backend.ctx
}

func (backend *Backend) Pool() *qpool.Q {
	if backend == nil {
		return nil
	}

	return backend.pool
}

func (backend *Backend) Mesh() tensor.ShardingMesh {
	if backend == nil {
		return tensor.ShardingMesh{}
	}

	return backend.mesh
}

func (backend *Backend) Devices() []*Device {
	if backend == nil {
		return nil
	}

	out := make([]*Device, len(backend.devices))
	copy(out, backend.devices)

	return out
}

func (backend *Backend) Device(deviceID DeviceID) (*Device, error) {
	if backend == nil {
		return nil, ErrDeviceNotFound
	}

	device, ok := backend.byID[deviceID]

	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrDeviceNotFound, deviceID)
	}

	return device, nil
}

func (backend *Backend) Memory(deviceID DeviceID) (tensor.Backend, error) {
	device, err := backend.Device(deviceID)

	if err != nil {
		return nil, err
	}

	if device.memory == nil {
		return nil, fmt.Errorf("%w: %s has no memory backend", ErrDeviceNotFound, deviceID)
	}

	return device.memory, nil
}

/*
DefaultComputeDevice prefers an accelerator with a live Executor, then host.
*/
func (backend *Backend) DefaultComputeDevice() DeviceID {
	if backend == nil {
		return DeviceID{Location: tensor.Host, Index: 0}
	}

	for _, location := range []tensor.Location{tensor.Metal, tensor.CUDA, tensor.XLA} {
		for _, device := range backend.devices {
			if device.id.Location != location || device.executor == nil {
				continue
			}

			return device.id
		}
	}

	return DeviceID{Location: tensor.Host, Index: 0}
}

/*
ResolveDevice maps a manifest device string to a discovered DeviceID.
*/
func (backend *Backend) ResolveDevice(raw string) (DeviceID, error) {
	if backend == nil {
		return DeviceID{}, ErrDeviceNotFound
	}

	trimmed := raw

	if trimmed == "" || trimmed == "auto" {
		return backend.DefaultComputeDevice(), nil
	}

	deviceID, err := ParseDeviceID(trimmed)

	if err != nil {
		return DeviceID{}, err
	}

	if _, err := backend.Device(deviceID); err != nil {
		return DeviceID{}, err
	}

	return deviceID, nil
}

/*
Execute routes one IR graph to the Executor on the requested device.
*/
func (backend *Backend) Execute(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
	on DeviceID,
) (map[string]tensor.Tensor, error) {
	device, err := backend.Device(on)

	if err != nil {
		return nil, err
	}

	if device.executor == nil {
		return nil, fmt.Errorf("%w: %s", ErrExecutorUnavailable, on)
	}

	return device.executor.Execute(ctx, graph, targets)
}

func (backend *Backend) Close() error {
	if backend == nil {
		return nil
	}

	var closeErr error

	backend.closeOnce.Do(func() {
		backend.cancel()

		for _, device := range backend.devices {
			if err := device.Close(); err != nil && closeErr == nil {
				closeErr = err
			}
		}
	})

	return closeErr
}
