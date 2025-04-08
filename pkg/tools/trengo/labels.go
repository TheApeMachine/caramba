package trengo

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Labels struct {
	client *Client
}

type Label struct {
	ID        int    `json:"id"`
	Name      string `json:"name"`
	Color     string `json:"color"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

type LabelList struct {
	Data []Label `json:"data"`
}

func NewLabels(client *Client) *Labels {
	return &Labels{client: client}
}

func (l *Labels) encode(artifact *datura.Artifact, v any) (err error) {
	data, err := json.Marshal(v)
	if err != nil {
		return errnie.Error(err)
	}

	datura.WithEncryptedPayload(data)(artifact)
	return nil
}

func (l *Labels) ListLabels(artifact *datura.Artifact) (err error) {
	resp, err := l.client.doRequest(http.MethodGet, "/labels", nil)
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return errnie.Error(err)
	}

	var labels LabelList
	if err = json.Unmarshal(body, &labels); err != nil {
		return errnie.Error(err)
	}

	return l.encode(artifact, labels)
}

func (l *Labels) GetLabel(artifact *datura.Artifact) (err error) {
	labelID := datura.GetMetaValue[int](artifact, "label_id")

	resp, err := l.client.doRequest(
		http.MethodGet,
		fmt.Sprintf("/labels/%d", labelID),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Body.Close()

	var label Label
	if err = json.NewDecoder(resp.Body).Decode(&label); err != nil {
		return errnie.Error(err)
	}

	return l.encode(artifact, label)
}

func (l *Labels) CreateLabel(artifact *datura.Artifact) (err error) {
	label := map[string]interface{}{
		"name":  datura.GetMetaValue[string](artifact, "name"),
		"color": datura.GetMetaValue[string](artifact, "color"),
	}

	resp, err := l.client.doRequest(http.MethodPost, "/labels", label)
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Body.Close()

	var newLabel Label
	if err = json.NewDecoder(resp.Body).Decode(&newLabel); err != nil {
		return errnie.Error(err)
	}

	return l.encode(artifact, newLabel)
}

func (l *Labels) UpdateLabel(artifact *datura.Artifact) (err error) {
	labelID := datura.GetMetaValue[int](artifact, "label_id")
	label := map[string]interface{}{
		"name":  datura.GetMetaValue[string](artifact, "name"),
		"color": datura.GetMetaValue[string](artifact, "color"),
	}

	resp, err := l.client.doRequest(
		http.MethodPut,
		fmt.Sprintf("/labels/%d", labelID),
		label,
	)
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Body.Close()

	var updatedLabel Label
	if err = json.NewDecoder(resp.Body).Decode(&updatedLabel); err != nil {
		return errnie.Error(err)
	}

	return l.encode(artifact, updatedLabel)
}

func (l *Labels) DeleteLabel(artifact *datura.Artifact) (err error) {
	labelID := datura.GetMetaValue[int](artifact, "label_id")

	resp, err := l.client.doRequest(
		http.MethodDelete,
		fmt.Sprintf("/labels/%d", labelID),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Body.Close()

	return nil
}
