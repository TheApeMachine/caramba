package compute

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/qpool"
)

func TestParseDeviceID(test *testing.T) {
	Convey("Given device id strings", test, func() {
		Convey("It should accept host aliases", func() {
			deviceID, err := ParseDeviceID("cpu")
			So(err, ShouldBeNil)
			So(deviceID, ShouldResemble, DeviceID{Location: tensor.Host, Index: 0})
		})

		Convey("It should parse indexed gpu ids", func() {
			deviceID, err := ParseDeviceID("metal:2")
			So(err, ShouldBeNil)
			So(deviceID, ShouldResemble, DeviceID{Location: tensor.Metal, Index: 2})
		})
	})
}

func TestNewBackend(test *testing.T) {
	Convey("Given a compute backend", test, func() {
		backend, err := NewBackend(context.Background(), nil)
		So(err, ShouldBeNil)

		defer func() {
			So(backend.Close(), ShouldBeNil)
		}()

		Convey("It should always discover host", func() {
			hostDevice, err := backend.Device(DeviceID{Location: tensor.Host, Index: 0})
			So(err, ShouldBeNil)
			So(hostDevice, ShouldNotBeNil)
		})

		Convey("It should reject unknown device ids", func() {
			_, err := backend.Device(DeviceID{Location: tensor.CUDA, Index: 9})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "device not found")
		})
	})
}

func TestNewBackend_Heterogeneous(test *testing.T) {
	Convey("Given discovery on a host with optional accelerators", test, func() {
		backend, err := NewBackend(context.Background(), qpool.NewQ(context.Background(), 1, 1, nil))
		So(err, ShouldBeNil)

		defer func() {
			So(backend.Close(), ShouldBeNil)
		}()

		Convey("It should keep host resident while metal memory is present", func() {
			hostDevice, err := backend.Device(DeviceID{Location: tensor.Host, Index: 0})
			So(err, ShouldBeNil)
			So(hostDevice, ShouldNotBeNil)

			metalDevice, err := backend.Device(DeviceID{Location: tensor.Metal, Index: 0})
			So(err, ShouldBeNil)
			So(metalDevice, ShouldNotBeNil)
		})
	})
}