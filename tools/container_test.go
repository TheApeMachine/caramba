package tools

// import (
// 	"context"
// 	"io"
// 	"os/exec"
// 	"testing"
// 	"time"

// 	. "github.com/smartystreets/goconvey/convey"
// )

// // cleanupContainer ensures no container with the given name exists
// func cleanupContainer() {
// 	// Stop the container first (ignore errors as it might not be running)
// 	stopCmd := exec.Command("docker", "stop", "caramba-terminal")
// 	_ = stopCmd.Run()

// 	// Remove the container with force
// 	rmCmd := exec.Command("docker", "rm", "-f", "caramba-terminal")
// 	_ = rmCmd.Run()

// 	// Wait a short moment to ensure Docker has fully processed the removal
// 	time.Sleep(100 * time.Millisecond)
// }

// func TestContainerInitialization(t *testing.T) {
// 	cleanupContainer()
// 	Convey("Given a new Container", t, func() {
// 		container := NewContainer()

// 		Reset(func() {
// 			if container.Conn != nil {
// 				container.Conn.Close()
// 			}
// 			cleanupContainer()
// 		})

// 		Convey("It should be properly initialized", func() {
// 			So(container, ShouldNotBeNil)
// 			So(container.builder, ShouldNotBeNil)
// 			So(container.runner, ShouldNotBeNil)
// 			So(container.Conn, ShouldBeNil)
// 		})

// 		Convey("It should have correct metadata", func() {
// 			So(container.Name(), ShouldEqual, "container")
// 			So(container.Description(), ShouldEqual, "Execute commands in an isolated Debian environment")
// 		})
// 	})
// }

// func TestContainerInitialize(t *testing.T) {
// 	cleanupContainer()
// 	Convey("Given a Container instance", t, func() {
// 		container := NewContainer()

// 		Reset(func() {
// 			if container.Conn != nil {
// 				container.Conn.Close()
// 			}
// 			cleanupContainer()
// 		})
// 	})
// }

// func TestContainerUse(t *testing.T) {
// 	cleanupContainer()
// 	Convey("Given an initialized Container", t, func() {
// 		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
// 		container := NewContainer()
// 		err := container.Start()
// 		So(err, ShouldBeNil)

// 		Reset(func() {
// 			cancel()
// 			if container.Conn != nil {
// 				container.Conn.Close()
// 			}
// 			cleanupContainer()
// 		})

// 		Convey("When executing commands", func() {
// 			Convey("It should handle invalid commands without crashing", func() {
// 				result := container.Use(ctx, map[string]any{})
// 				So(result, ShouldEqual, "error: invalid command parameter")
// 			})

// 			Convey("It should execute valid commands successfully", func() {
// 				result := container.Use(ctx, map[string]any{
// 					"command": "echo 'hello world'",
// 				})
// 				So(result, ShouldContainSubstring, "hello world")
// 			})

// 			Convey("It should maintain state between commands", func() {
// 				_ = container.Use(ctx, map[string]any{
// 					"command": "touch testfile.txt",
// 				})
// 				result := container.Use(ctx, map[string]any{
// 					"command": "ls testfile.txt",
// 				})
// 				So(result, ShouldContainSubstring, "testfile.txt")
// 			})
// 		})
// 	})
// }

// func TestContainerConnect(t *testing.T) {
// 	cleanupContainer()
// 	Convey("Given a Container instance", t, func() {
// 		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
// 		container := NewContainer()

// 		Reset(func() {
// 			cancel()
// 			if container.Conn != nil {
// 				container.Conn.Close()
// 			}
// 			cleanupContainer()
// 		})

// 		Convey("When connecting with a bridge", func() {
// 			mockBridge := &mockReadWriteCloser{}

// 			Convey("It should establish a connection", func() {
// 				err := container.Connect(ctx, mockBridge)
// 				So(err, ShouldBeNil)
// 				So(container.Conn, ShouldNotBeNil)
// 			})

// 			Convey("It should reuse the bridge for subsequent connections", func() {
// 				err := container.Connect(ctx, mockBridge)
// 				So(err, ShouldBeNil)
// 				firstConn := container.Conn

// 				err = container.Connect(ctx, mockBridge)
// 				So(err, ShouldBeNil)
// 				So(container.Conn, ShouldEqual, firstConn)
// 			})
// 		})
// 	})
// }

// // mockReadWriteCloser implements io.ReadWriteCloser for testing
// type mockReadWriteCloser struct {
// 	readData  []byte
// 	writeData []byte
// }

// func (m *mockReadWriteCloser) Read(p []byte) (n int, err error) {
// 	if len(m.readData) == 0 {
// 		return 0, io.EOF
// 	}
// 	n = copy(p, m.readData)
// 	m.readData = m.readData[n:]
// 	return n, nil
// }

// func (m *mockReadWriteCloser) Write(p []byte) (n int, err error) {
// 	m.writeData = append(m.writeData, p...)
// 	return len(p), nil
// }

// func (m *mockReadWriteCloser) Close() error {
// 	return nil
// }
