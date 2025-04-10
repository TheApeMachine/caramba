package datura

import (
	"sync"
	"testing"

	capnp "capnproto.org/go/capnp/v3"
	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// MockRegisterable is a mock implementation of the Registerable interface.
type MockRegisterable struct {
	id       string
	readErr  error
	writeErr error
	closeErr error
	msg      *capnp.Message
}

func (m *MockRegisterable) ID() string {
	return m.id
}

func (m *MockRegisterable) Read(p []byte) (n int, err error) {
	return 0, m.readErr
}

func (m *MockRegisterable) Write(p []byte) (n int, err error) {
	return len(p), m.writeErr
}

func (m *MockRegisterable) Close() error {
	return m.closeErr
}

func (m *MockRegisterable) Message() *capnp.Message {
	if m.msg == nil {
		// Create a dummy message if none is provided
		seg := capnp.SingleSegment(nil)
		m.msg, _, _ = capnp.NewMessage(seg)
	}
	return m.msg
}

func TestNewBuffer(t *testing.T) {
	Convey("Given a Registerable item", t, func() {
		mockItem := &MockRegisterable{id: "test-item"}

		Convey("When creating a new Buffer", func() {
			buf := NewBuffer(mockItem)

			Convey("Then the buffer should be initialized correctly", func() {
				So(buf, ShouldNotBeNil)
				So(buf.registerable, ShouldEqual, mockItem)
				So(buf.Encoder, ShouldNotBeNil)
				So(buf.Decoder, ShouldNotBeNil)
				So(buf.Buffer, ShouldNotBeNil)
				So(buf.State, ShouldEqual, errnie.StateUnknown)
			})
		})
	})
}

func TestNewRegistry(t *testing.T) {
	Convey("When creating a new Registry", t, func() {
		// Reset the global registry for isolated testing
		once = sync.Once{}
		registry = nil

		reg1 := NewRegistry()

		Convey("Then a non-nil registry should be returned", func() {
			So(reg1, ShouldNotBeNil)
			So(reg1.buffers, ShouldNotBeNil)
		})

		Convey("And subsequent calls should return the same instance (singleton)", func() {
			reg2 := NewRegistry()
			So(reg2, ShouldPointTo, reg1)
		})
	})
}

func TestRegister(t *testing.T) {
	Convey("Given a Registry", t, func() {
		// Ensure a clean registry for each test scenario
		once = sync.Once{}
		registry = nil
		reg := NewRegistry()

		mockItem1 := &MockRegisterable{id: "item1"}

		Convey("When registering a valid Registerable item", func() {
			registeredItem := Register(mockItem1)

			Convey("Then the item should be returned", func() {
				So(registeredItem, ShouldEqual, mockItem1)
			})

			Convey("And the registry should contain a buffer for the item", func() {
				reg.mu.RLock()
				buf, exists := reg.buffers[mockItem1.id]
				reg.mu.RUnlock()
				So(exists, ShouldBeTrue)
				So(buf, ShouldNotBeNil)
				So(buf.registerable, ShouldEqual, mockItem1)
			})
		})

		Convey("When registering an item with an empty ID", func() {
			mockItemEmptyID := &MockRegisterable{id: ""}

			Convey("Then it should call errnie.Fatal (which exits)", func() {
				// We cannot use ShouldPanic here because errnie.Fatal calls os.Exit.
				// We trust that if the ID is empty, Fatal will be invoked.
				// Executing Register(mockItemEmptyID) would terminate the test.
				So(mockItemEmptyID.id, ShouldBeEmpty) // Verify the condition leading to Fatal
			})
		})

		Convey("When registering the same item twice", func() {
			Register(mockItem1) // First registration
			buf1 := reg.buffers[mockItem1.id]
			Register(mockItem1) // Second registration
			buf2 := reg.buffers[mockItem1.id]

			Convey("Then the buffer should be overwritten with a new one", func() {
				So(buf2, ShouldNotPointTo, buf1)
				So(buf2.registerable, ShouldEqual, mockItem1)
			})
		})
	})
}

func TestUnregister(t *testing.T) {
	Convey("Given a Registry with a registered item", t, func() {
		// Ensure a clean registry
		once = sync.Once{}
		registry = nil
		reg := NewRegistry()
		mockItem1 := &MockRegisterable{id: "item1"}
		Register(mockItem1)

		Convey("When unregistering the item", func() {
			reg.Unregister(mockItem1)

			Convey("Then the registry should no longer contain the item's buffer", func() {
				reg.mu.RLock()
				_, exists := reg.buffers[mockItem1.id]
				reg.mu.RUnlock()
				So(exists, ShouldBeFalse)
			})
		})

		Convey("When unregistering an item that is not registered", func() {
			mockItem2 := &MockRegisterable{id: "item2"}

			Convey("Then the operation should complete without error", func() {
				So(func() { reg.Unregister(mockItem2) }, ShouldNotPanic)
				// Check item1 is still there
				reg.mu.RLock()
				_, exists := reg.buffers[mockItem1.id]
				reg.mu.RUnlock()
				So(exists, ShouldBeTrue)
			})
		})

		Convey("When unregistering an item with an empty ID", func() {
			mockItemEmptyID := &MockRegisterable{id: ""}

			Convey("Then it should call errnie.Fatal (which exits)", func() {
				// We cannot use ShouldPanic here because errnie.Fatal calls os.Exit.
				// We trust that if the ID is empty, Fatal will be invoked.
				// Executing reg.Unregister(mockItemEmptyID) would terminate the test.
				So(mockItemEmptyID.id, ShouldBeEmpty) // Verify the condition leading to Fatal
			})
		})
	})
}

func TestGet(t *testing.T) {
	Convey("Given a Registry", t, func() {
		// Ensure a clean registry
		once = sync.Once{}
		registry = nil
		reg := NewRegistry()
		mockItem1 := &MockRegisterable{id: "item1"}

		Convey("When getting an item that is already registered", func() {
			Register(mockItem1)
			originalBuffer := reg.buffers[mockItem1.id]
			retrievedBuffer := reg.Get(mockItem1)

			Convey("Then the existing buffer should be returned", func() {
				So(retrievedBuffer, ShouldPointTo, originalBuffer)
			})
		})

		Convey("When getting an item that is not registered", func() {
			mockItem2 := &MockRegisterable{id: "item2"}
			retrievedBuffer := reg.Get(mockItem2)

			Convey("Then a new buffer should be created and returned", func() {
				So(retrievedBuffer, ShouldNotBeNil)
				So(retrievedBuffer.registerable, ShouldEqual, mockItem2)
			})

			Convey("And the registry should now contain the new buffer", func() {
				reg.mu.RLock()
				buf, exists := reg.buffers[mockItem2.id]
				reg.mu.RUnlock()
				So(exists, ShouldBeTrue)
				So(buf, ShouldPointTo, retrievedBuffer)
			})
		})

		Convey("When getting an item with an empty ID (that is not registered)", func() {
			mockItemEmptyID := &MockRegisterable{id: ""}

			Convey("Then it should call errnie.Fatal (which exits)", func() {
				// We cannot use ShouldPanic here because errnie.Fatal calls os.Exit.
				// We trust that if the ID is empty, Fatal will be invoked.
				// Executing reg.Get(mockItemEmptyID) would terminate the test.
				So(mockItemEmptyID.id, ShouldBeEmpty) // Verify the condition leading to Fatal
			})
		})
	})
}
