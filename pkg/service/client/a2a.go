package client

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"runtime"
	"time"

	"github.com/goccy/go-json"

	fiberClient "github.com/gofiber/fiber/v3/client"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

// A2AClient provides methods for communicating with an A2A server
type A2AClient struct {
	conn *fiberClient.Client
}

type A2AClientOption func(*A2AClient)

// NewA2AClient creates a new A2A client to communicate with an A2A server
func NewA2AClient(opts ...A2AClientOption) *A2AClient {
	a2a := &A2AClient{}

	for _, opt := range opts {
		opt(a2a)
	}

	return a2a
}

// SendTask sends a task to the A2A server
func (client *A2AClient) SendTask(
	req *task.TaskRequest, out io.Writer,
) (*task.TaskResponse, error) {
	resp, err := client.Post("/rpc", req)

	if err != nil {
		return nil, errnie.New(errnie.WithError(err))
	}

	collector := task.NewTaskCollector(req, out)
	reader := bufio.NewReader(bytes.NewReader(resp.Body()))

	// Read the response stream line by line
	for {
		line, err := reader.ReadBytes('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, errnie.New(errnie.WithError(err))
		}

		// Skip empty lines
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}

		if _, err = collector.Write(line); err != nil {
			return nil, errnie.New(errnie.WithError(err))
		}
	}

	return collector.Response(), nil
}

func (client *A2AClient) Post(
	url string, body any,
) (*fiberClient.Response, error) {
	errnie.Debug("POST", "url", url)

	return client.conn.Post(
		url,
		fiberClient.Config{
			Header: map[string]string{
				"Content-Type": "application/json",
			},
			Body:      body,
			UserAgent: getUserAgent(),
			Timeout:   10 * time.Second,
		},
	)
}

func getUserAgent() string {
	return fmt.Sprintf("%s/%s (%s; %s; %s)",
		tweaker.Value[string]("settings.app.name"),
		tweaker.Value[string]("settings.app.version"),
		runtime.GOOS,
		runtime.GOARCH,
		runtime.Version(),
	)
}

func WithBaseURL(baseURL string) A2AClientOption {
	return func(a2a *A2AClient) {
		a2a.conn = fiberClient.New()
		a2a.conn.SetBaseURL(baseURL)
		a2a.conn.SetJSONMarshal(json.Marshal)
		a2a.conn.SetJSONUnmarshal(json.Unmarshal)
	}
}
