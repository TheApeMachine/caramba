package container

// import (
// 	"context"
// 	"os/exec"
// 	"strings"
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

// func TestNewRunner(t *testing.T) {
// 	Convey("When creating a new Runner", t, func() {
// 		runner := NewRunner()

// 		Convey("Then it should be properly initialized", func() {
// 			So(runner, ShouldNotBeNil)
// 			So(runner.client, ShouldNotBeNil)
// 			So(runner.containerID, ShouldBeEmpty)
// 		})
// 	})
// }

// func TestRunContainer(t *testing.T) {
// 	cleanupContainer()
// 	Convey("Given a Runner instance", t, func() {
// 		ctx := context.Background()
// 		runner := NewRunner()

// 		Reset(func() {
// 			if runner.containerID != "" {
// 				runner.StopContainer(ctx)
// 			}
// 			cleanupContainer()
// 		})

// 		Convey("When running a container for the first time", func() {
// 			conn, err := runner.RunContainer(ctx, DefaultImageName)
// 			defer func() {
// 				if conn != nil {
// 					conn.Close()
// 				}
// 			}()

// 			Convey("Then it should create and connect successfully", func() {
// 				So(err, ShouldBeNil)
// 				So(conn, ShouldNotBeNil)
// 				So(runner.containerID, ShouldNotBeEmpty)
// 			})

// 			if conn != nil {
// 				Convey("And the container should be interactive", func() {
// 					// Write a command with a clear end marker
// 					_, err := conn.Write([]byte("echo 'test' && echo 'DONE'\n"))
// 					So(err, ShouldBeNil)

// 					// Read the response with timeout
// 					buf := make([]byte, 4096)
// 					done := make(chan struct{})
// 					var output string

// 					go func() {
// 						defer close(done)
// 						for !strings.Contains(output, "DONE") {
// 							n, err := conn.Read(buf)
// 							if err != nil {
// 								return
// 							}
// 							if n > 0 {
// 								output += string(buf[:n])
// 							}
// 						}
// 					}()

// 					select {
// 					case <-done:
// 						So(output, ShouldContainSubstring, "test")
// 					case <-time.After(5 * time.Second):
// 						t.Fatal("timeout waiting for container response")
// 					}
// 				})
// 			}
// 		})

// 		Convey("When running a container with an invalid image", func() {
// 			conn, err := runner.RunContainer(ctx, "nonexistent:image")
// 			defer func() {
// 				if conn != nil {
// 					conn.Close()
// 				}
// 			}()

// 			Convey("Then it should return an error", func() {
// 				So(err, ShouldNotBeNil)
// 				So(conn, ShouldBeNil)
// 			})
// 		})
// 	})
// }

// func TestExecuteCommand(t *testing.T) {
// 	cleanupContainer()
// 	Convey("Given a running container", t, func() {
// 		ctx := context.Background()
// 		runner := NewRunner()
// 		conn, err := runner.RunContainer(ctx, DefaultImageName)
// 		So(err, ShouldBeNil)
// 		defer func() {
// 			if conn != nil {
// 				conn.Close()
// 			}
// 		}()

// 		Reset(func() {
// 			if runner.containerID != "" {
// 				runner.StopContainer(ctx)
// 			}
// 			cleanupContainer()
// 		})

// 		Convey("When executing a valid command", func() {
// 			output := runner.ExecuteCommand(ctx, []string{"echo 'hello world'"})

// 			Convey("Then it should return the expected output", func() {
// 				So(output, ShouldNotBeNil)
// 				So(string(output), ShouldContainSubstring, "hello world")
// 			})
// 		})

// 		Convey("When executing multiple commands", func() {
// 			_ = runner.ExecuteCommand(ctx, []string{"touch testfile"})
// 			second := runner.ExecuteCommand(ctx, []string{"ls testfile"})

// 			Convey("Then it should maintain state between commands", func() {
// 				So(string(second), ShouldContainSubstring, "testfile")
// 			})
// 		})
// 	})
// }

// func TestStopContainer(t *testing.T) {
// 	cleanupContainer()
// 	Convey("Given a running container", t, func() {
// 		ctx := context.Background()
// 		runner := NewRunner()
// 		conn, err := runner.RunContainer(ctx, DefaultImageName)
// 		So(err, ShouldBeNil)
// 		defer func() {
// 			if conn != nil {
// 				conn.Close()
// 			}
// 		}()

// 		Reset(func() {
// 			cleanupContainer()
// 		})

// 		Convey("When stopping the container", func() {
// 			err := runner.StopContainer(ctx)

// 			Convey("Then it should stop successfully", func() {
// 				So(err, ShouldBeNil)
// 			})

// 			Convey("And subsequent commands should fail", func() {
// 				output := runner.ExecuteCommand(ctx, []string{"echo 'test'"})
// 				So(output, ShouldBeNil)
// 			})
// 		})
// 	})
// }
