package errnie

import (
	"bytes"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

/*
elasticPostWriter writes one JSON log line per Write call to Elasticsearch _doc API.
*/
type elasticPostWriter struct {
	endpoint   string
	httpClient *http.Client
	username   string
	password   string
}

func newElasticPostWriter(baseURL, index, username, password string) (*elasticPostWriter, error) {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	idx := strings.TrimSpace(index)

	if base == "" || idx == "" {
		return nil, fmt.Errorf("elasticsearch: url and index required")
	}

	endpoint, err := url.JoinPath(base, idx, "_doc")

	if err != nil {
		return nil, err
	}

	return &elasticPostWriter{
		endpoint: endpoint,
		httpClient: &http.Client{
			Timeout: 15 * time.Second,
		},
		username: strings.TrimSpace(username),
		password: password,
	}, nil
}

/*
Write.
*/
func (sink *elasticPostWriter) Write(payload []byte) (int, error) {
	if len(payload) == 0 {
		return 0, nil
	}

	request, err := http.NewRequest(http.MethodPost, sink.endpoint, bytes.NewReader(payload))

	if err != nil {
		return 0, err
	}

	request.Header.Set("Content-Type", "application/json")

	if sink.username != "" || sink.password != "" {
		request.SetBasicAuth(sink.username, sink.password)
	}

	response, err := sink.httpClient.Do(request)

	if err != nil {
		return 0, err
	}

	defer response.Body.Close()

	if response.StatusCode >= http.StatusMultipleChoices {
		return 0, fmt.Errorf("elasticsearch: %s", response.Status)
	}

	return len(payload), nil
}
