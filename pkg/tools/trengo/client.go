package trengo

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Client struct {
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

	return client
}

func (c *Client) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)
	}()

	return out
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
