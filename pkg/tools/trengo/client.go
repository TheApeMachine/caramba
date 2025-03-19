package trengo

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Client struct {
	buffer   *stream.Buffer
	baseURL  string
	apiToken string
	http     *http.Client
	tickets  *Tickets
	labels   *Labels
}

func NewClient() *Client {
	client := &Client{
		baseURL:  "https://app.trengo.com/api/v2",
		apiToken: os.Getenv("TRENGO_API_TOKEN"),
		http:     &http.Client{},
	}

	client.tickets = NewTickets(client)
	client.labels = NewLabels(client)

	client.buffer = stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
		errnie.Debug("trengo.Client.buffer")

		switch artifact.Role() {
		case uint32(datura.ArtifactRoleListTickets):
			return client.tickets.ListTickets(artifact)
		case uint32(datura.ArtifactRoleCreateTicket):
			return client.tickets.CreateTicket(artifact)
		case uint32(datura.ArtifactRoleAssignTicket):
			return client.tickets.AssignTicket(artifact)
		case uint32(datura.ArtifactRoleCloseTicket):
			return client.tickets.CloseTicket(artifact)
		case uint32(datura.ArtifactRoleReopenTicket):
			return client.tickets.ReopenTicket(artifact)
		case uint32(datura.ArtifactRoleListLabels):
			return client.labels.ListLabels(artifact)
		case uint32(datura.ArtifactRoleGetLabel):
			return client.labels.GetLabel(artifact)
		case uint32(datura.ArtifactRoleCreateLabel):
			return client.labels.CreateLabel(artifact)
		case uint32(datura.ArtifactRoleUpdateLabel):
			return client.labels.UpdateLabel(artifact)
		case uint32(datura.ArtifactRoleDeleteLabel):
			return client.labels.DeleteLabel(artifact)
		}

		return nil
	})

	return client
}

func (client *Client) Read(p []byte) (n int, err error) {
	errnie.Debug("trengo.Client.Read")
	return client.buffer.Read(p)
}

func (client *Client) Write(p []byte) (n int, err error) {
	errnie.Debug("trengo.Client.Write")
	return client.buffer.Write(p)
}

func (client *Client) Close() error {
	errnie.Debug("trengo.Client.Close")
	return client.buffer.Close()
}

func (client *Client) doRequest(method, path string, body interface{}) (*http.Response, error) {
	var bodyReader *bytes.Reader

	if body != nil {
		bodyBytes, err := json.Marshal(body)
		if err != nil {
			return nil, errnie.Error(err)
		}
		bodyReader = bytes.NewReader(bodyBytes)
	}

	req, err := http.NewRequest(method, fmt.Sprintf("%s%s", client.baseURL, path), bodyReader)
	if err != nil {
		return nil, errnie.Error(err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", client.apiToken))
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.http.Do(req)
	if err != nil {
		return nil, errnie.Error(err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("trengo API error: %s", resp.Status)
	}

	return resp, nil
}
