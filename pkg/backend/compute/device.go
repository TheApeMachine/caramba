package compute

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/runtime"
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
