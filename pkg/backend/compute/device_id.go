package compute

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/tensor"
)

var ErrUnsupportedLocation = fmt.Errorf("compute: unsupported location")

/*
DeviceID names one execution context in the Backend mesh.
*/
type DeviceID struct {
	Location tensor.Location
	Index    int
}

func (deviceID DeviceID) String() string {
	if deviceID.Index == 0 {
		return string(deviceID.Location)
	}

	return fmt.Sprintf("%s:%d", deviceID.Location, deviceID.Index)
}

/*
ParseDeviceID resolves manifest strings such as "host", "metal:1", or "cuda:0".
*/
func ParseDeviceID(raw string) (DeviceID, error) {
	trimmed := strings.ToLower(strings.TrimSpace(raw))

	if trimmed == "" || trimmed == "cpu" || trimmed == "host" {
		return DeviceID{Location: tensor.Host, Index: 0}, nil
	}

	locationPart, indexPart, hasIndex := strings.Cut(trimmed, ":")

	location, err := locationFromName(locationPart)

	if err != nil {
		return DeviceID{}, err
	}

	if !hasIndex {
		return DeviceID{Location: location, Index: 0}, nil
	}

	var index int

	if _, err := fmt.Sscanf(indexPart, "%d", &index); err != nil || index < 0 {
		return DeviceID{}, fmt.Errorf("compute: invalid device index %q", indexPart)
	}

	return DeviceID{Location: location, Index: index}, nil
}

func locationFromName(name string) (tensor.Location, error) {
	switch name {
	case "host", "cpu":
		return tensor.Host, nil
	case "metal":
		return tensor.Metal, nil
	case "cuda":
		return tensor.CUDA, nil
	case "xla":
		return tensor.XLA, nil
	case "network":
		return tensor.Network, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrUnsupportedLocation, name)
	}
}
