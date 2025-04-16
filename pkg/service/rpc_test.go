package service

import (
	"context"
	"errors"
	"io"
	"net"
	"testing"
	"time"

	"github.com/gofiber/fiber/v3"
	. "github.com/smartystreets/goconvey/convey"
	"github.com/sourcegraph/jsonrpc2"
	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/caramba/pkg/registry"
	"github.com/theapemachine/caramba/pkg/task"
)

const (
	taskSendMethod = "tasks/send"
	localhost      = "127.0.0.1:0" // Use 127.0.0.1 for explicit localhost
)

// MockAgent implements the agent.Agent interface for testing
type MockAgent struct {
	HandleTaskFunc func(ctx fiber.Ctx, req *task.TaskRequest) error
	CardFunc       func() *agent.Card
	AddWriterFunc  func(w io.Writer)
	TaskHandled    bool // Flag to check if HandleTask was called
}

func NewMockAgent() *MockAgent {
	return &MockAgent{
		HandleTaskFunc: func(ctx fiber.Ctx, req *task.TaskRequest) error {
			// Default mock behavior: Mark as handled and return nil
			return nil
		},
		CardFunc: func() *agent.Card {
			return &agent.Card{Name: "mock-agent"} // Provide a default card
		},
	}
}

func (a *MockAgent) HandleTask(ctx fiber.Ctx, req *task.TaskRequest) error {
	if a.HandleTaskFunc != nil {
		a.TaskHandled = true // Set flag when called
		return a.HandleTaskFunc(ctx, req)
	}
	return errors.New("HandleTaskFunc not set")
}

func (a *MockAgent) Card() *agent.Card {
	if a.CardFunc != nil {
		return a.CardFunc()
	}
	return nil
}

func (a *MockAgent) AddWriter(w io.Writer) {
	if a.AddWriterFunc != nil {
		a.AddWriterFunc(w)
	}
}

func TestNewHandler(t *testing.T) {
	Convey("Given a new RPC handler", t, func() {
		mockAgent := NewMockAgent()
		handler := NewHandler(mockAgent, registry.NewMockRegistry())

		Convey("It should create a valid handler", func() {
			So(handler, ShouldNotBeNil)
			So(handler.agent, ShouldEqual, mockAgent)
		})
	})
}

// setupTestServer sets up a listener and handler for RPC tests
func setupTestServer(t *testing.T, h *Handler) (net.Listener, string) {
	listener, err := net.Listen("tcp", localhost)
	So(err, ShouldBeNil)
	addr := listener.Addr().String()

	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				if errors.Is(err, net.ErrClosed) {
					return // Listener closed, exit goroutine
				}
				t.Logf("Listener accept error: %v", err)
				return
			}
			stream := jsonrpc2.NewBufferedStream(conn, jsonrpc2.VSCodeObjectCodec{})
			jsonrpc2.NewConn(context.Background(), stream, jsonrpc2.HandlerWithError(h.Handle))
		}
	}()

	return listener, addr
}

// setupTestClient connects to the test server
func setupTestClient(t *testing.T, addr string) *jsonrpc2.Conn {
	conn, err := net.Dial("tcp", addr)
	So(err, ShouldBeNil)
	stream := jsonrpc2.NewBufferedStream(conn, jsonrpc2.VSCodeObjectCodec{})
	clientConn := jsonrpc2.NewConn(context.Background(), stream, jsonrpc2.HandlerWithError(func(context.Context, *jsonrpc2.Conn, *jsonrpc2.Request) (any, error) {
		return nil, nil // Dummy handler for server-to-client messages (if any)
	}))
	return clientConn
}

func TestHandle(t *testing.T) {
	Convey("Given a new RPC handler and a running test server/client", t, func() {
		mockAgent := NewMockAgent()
		mockRegistry := registry.NewMockRegistry()
		handler := NewHandler(mockAgent, mockRegistry)
		listener, addr := setupTestServer(t, handler)
		clientConn := setupTestClient(t, addr)

		// Ensure resources are cleaned up
		Reset(func() {
			clientConn.Close()
			listener.Close()
		})

		// Convey("When handling a valid task/send request with multiple part types", func() {
		// 	testTask := task.NewTask(
		// 		task.WithMessages(task.Message{
		// 			Role: task.RoleUser,
		// 			Parts: []task.Part{
		// 				&task.TextPart{Type: "text", Text: "Hello world"},
		// 				&task.FilePart{Type: "file", File: task.FileContent{Name: "test.txt"}},
		// 			},
		// 		}),
		// 	)

		// 	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		// 	defer cancel()

		// 	var result task.Task // Expect the result to be the task itself
		// 	err := clientConn.Call(ctx, taskSendMethod, testTask /* Pass Task directly */, &result)

		// 	Convey("It should process the request successfully", func() {
		// 		So(err, ShouldBeNil)
		// 		So(mockAgent.TaskHandled, ShouldBeTrue) // Verify agent method was called
		// 		So(result.ID, ShouldEqual, testTask.ID) // Verify returned task ID matches
		// 		So(len(result.History), ShouldEqual, 1)
		// 		So(len(result.History[0].Parts), ShouldEqual, 2)
		// 	})
		// })

		Convey("When handling invalid requests", func() {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			// Convey("It should handle requests with nil params correctly", func() {
			// 	reqID := jsonrpc2.ID{
			// 		Num: 1,
			// 	}
			// 	jReq := &jsonrpc2.Request{
			// 		Method: taskSendMethod,
			// 		ID:     reqID,
			// 		Params: nil, // Explicitly nil
			// 	}
			// 	var resp jsonrpc2.Response // Capture raw response
			// 	err := clientConn.Call(ctx, jReq.Method, jReq.Params, &resp)

			// 	// We expect an error response from the server
			// 	So(err, ShouldNotBeNil)
			// 	// Check for the error that occurs when validation fails due to empty params
			// 	So(err.Error(), ShouldContainSubstring, "invalid task request: task ID is required")
			// 	rpcErr, ok := err.(*jsonrpc2.Error)
			// 	So(ok, ShouldBeTrue)
			// 	So(rpcErr.Code, ShouldEqual, jsonrpc2.CodeInvalidParams)
			// })

			// Convey("It should handle requests with missing task ID correctly", func() {
			// 	invalidTask := task.Task{ /* Missing ID */ }
			// 	var result any
			// 	err := clientConn.Call(ctx, taskSendMethod, invalidTask, &result)

			// 	So(err, ShouldNotBeNil)
			// 	So(err.Error(), ShouldContainSubstring, "invalid task request: task ID is required")
			// 	rpcErr, ok := err.(*jsonrpc2.Error)
			// 	So(ok, ShouldBeTrue)
			// 	So(rpcErr.Code, ShouldEqual, jsonrpc2.CodeInvalidParams)
			// })

			Convey("It should handle requests for unknown methods correctly", func() {
				var result any
				err := clientConn.Call(ctx, "unknown/method", map[string]string{"data": "foo"}, &result)

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "method not found: unknown/method")
				rpcErr, ok := err.(*jsonrpc2.Error)
				So(ok, ShouldBeTrue)
				So(rpcErr.Code, ShouldEqual, jsonrpc2.CodeMethodNotFound)
			})

			// Note: Testing for req == nil cannot be done via client call, only direct handler call if needed.
		})
	})
}

func TestStartRPCServer(t *testing.T) {
	Convey("Given an A2A instance and a request to start the RPC server", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		// Use assert for simpler nil checks
		mockAgent := NewMockAgent()
		So(mockAgent, ShouldNotBeNil)

		a2a := &A2A{
			agent: mockAgent,
		}

		// Create a listener to get a free port
		l, err := net.Listen("tcp", localhost)
		So(err, ShouldBeNil)
		addr := l.Addr().String()
		So(l.Close(), ShouldBeNil) // Close the temp listener immediately

		serverErr := make(chan error, 1)
		serverDone := make(chan struct{}) // To signal server goroutine completion

		go func() {
			err := a2a.StartRPCServer(ctx, addr)
			if err != nil && !errors.Is(err, net.ErrClosed) {
				serverErr <- err
			}
			close(serverDone)
		}()

		// Wait briefly for the server to potentially start listening
		// A more robust approach might involve polling the address.
		time.Sleep(100 * time.Millisecond)

		Reset(func() {
			cancel() // Cancel context to signal server shutdown
			// No direct way to close the listener started in StartRPCServer, relies on context cancellation or Accept error.
			<-serverDone // Wait for server goroutine to finish
		})

		Convey("When a client connects and sends a valid task request", func() {
			clientConn := setupTestClient(t, addr)
			defer clientConn.Close()

			testTask := task.NewTask(task.WithMessages(task.NewUserMessage("test", "data")))
			var result task.Task

			callCtx, callCancel := context.WithTimeout(ctx, 5*time.Second)
			defer callCancel()

			err = clientConn.Call(callCtx, "tasks/send", testTask, &result)

			So(err, ShouldBeNil)
			So(mockAgent.TaskHandled, ShouldBeTrue)
			So(result.ID, ShouldEqual, testTask.ID)
		})

		Convey("When the server start fails (e.g., port already in use)", func() {
			// Start another listener on the same port to cause an error
			l2, err := net.Listen("tcp", addr)
			if err != nil {
				t.Logf("Could not listen on %s for error test: %v", addr, err)
				return // Skip if we can't set up the conflicting listener
			}
			defer l2.Close()

			// Start a second server instance destined to fail
			a2aFail := &A2A{agent: NewMockAgent()}
			// Use context.Background() for this test instance as it's not the primary server we're testing for cancellation
			startErr := a2aFail.StartRPCServer(context.Background(), addr) // This call will block until error
			So(startErr, ShouldNotBeNil)
			So(startErr.Error(), ShouldContainSubstring, "bind: address already in use") // Or similar OS-specific error
		})

		// Check if the main server goroutine encountered an unexpected error
		select {
		case err := <-serverErr:
			// Use Fatalf here as an unexpected server error is critical
			t.Fatalf("Unexpected RPC server error: %v", err)
		default:
			// No error, proceed
		}
	})
}
