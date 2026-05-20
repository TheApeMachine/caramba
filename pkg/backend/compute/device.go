package compute

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/runtime"
	devicemetal "github.com/theapemachine/caramba/pkg/backend/device/metal"
	"github.com/theapemachine/manifesto/tensor"
)

/*
Device is one mesh cell: resident memory and the Executor that runs IR
on that memory. Backend holds every discovered Device concurrently.
*/
type Device struct {
	id       DeviceID
	memory   tensor.Backend
	executor runtime.Executor
}

func (device *Device) ID() DeviceID {
	if device == nil {
		return DeviceID{}
	}

	return device.id
}

func (device *Device) Location() tensor.Location {
	return device.id.Location
}

func (device *Device) Memory() tensor.Backend {
	if device == nil {
		return nil
	}

	return device.memory
}

func (device *Device) Executor() runtime.Executor {
	if device == nil {
		return nil
	}

	return device.executor
}

func (device *Device) Close() error {
	if device == nil {
		return nil
	}

	if device.executor != nil {
		return device.executor.Close()
	}

	if device.memory != nil {
		return device.memory.Close()
	}

	return nil
}

func discoverDevices() ([]*Device, error) {
	devices := make([]*Device, 0, 4)
	devices = append(devices, newHostDevice(0))
	devices = appendMetalDevice(devices)

	return devices, nil
}

func newHostDevice(index int) *Device {
	hostMemory := tensor.NewHostBackend()

	return &Device{
		id:       DeviceID{Location: tensor.Host, Index: index},
		memory:   hostMemory,
		executor: runtime.NewHost(hostMemory),
	}
}

func appendMetalDevice(devices []*Device) []*Device {
	metalMemory, err := devicemetal.NewBackend()

	if err != nil {
		return devices
	}

	return append(devices, &Device{
		id:     DeviceID{Location: tensor.Metal, Index: 0},
		memory: metalMemory,
		// Metal graph Executor lands with device/metal kernel dispatch.
		executor: nil,
	})
}

func indexDevices(devices []*Device) map[DeviceID]*Device {
	byID := make(map[DeviceID]*Device, len(devices))

	for _, device := range devices {
		byID[device.id] = device
	}

	return byID
}

func buildMesh(devices []*Device) tensor.ShardingMesh {
	locations := make([]tensor.Location, len(devices))

	for index, device := range devices {
		locations[index] = device.id.Location
	}

	return tensor.ShardingMesh{
		Devices:   locations,
		Shape:     []int{len(devices)},
		AxisNames: []string{"device"},
	}
}
