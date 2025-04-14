package trengo

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

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

func (l *Labels) ListLabels() (labels LabelList, err error) {
	resp, err := l.client.doRequest(http.MethodGet, "/labels", nil)
	if err != nil {
		return LabelList{}, errnie.New(errnie.WithError(err))
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return LabelList{}, errnie.New(errnie.WithError(err))
	}

	if err = json.Unmarshal(body, &labels); err != nil {
		return LabelList{}, errnie.New(errnie.WithError(err))
	}

	return labels, nil
}

func (l *Labels) GetLabel(labelID int) (label Label, err error) {
	resp, err := l.client.doRequest(
		http.MethodGet,
		fmt.Sprintf("/labels/%d", labelID),
		nil,
	)
	if err != nil {
		return Label{}, errnie.New(errnie.WithError(err))
	}
	defer resp.Body.Close()

	if err = json.NewDecoder(resp.Body).Decode(&label); err != nil {
		return Label{}, errnie.New(errnie.WithError(err))
	}

	return label, nil
}

func (l *Labels) CreateLabel(label Label) (newLabel Label, err error) {
	resp, err := l.client.doRequest(http.MethodPost, "/labels", label)
	if err != nil {
		return Label{}, errnie.New(errnie.WithError(err))
	}
	defer resp.Body.Close()

	if err = json.NewDecoder(resp.Body).Decode(&newLabel); err != nil {
		return Label{}, errnie.New(errnie.WithError(err))
	}

	return newLabel, nil
}

func (l *Labels) UpdateLabel(labelID int, label Label) (updatedLabel Label, err error) {
	resp, err := l.client.doRequest(
		http.MethodPut,
		fmt.Sprintf("/labels/%d", labelID),
		label,
	)
	if err != nil {
		return Label{}, errnie.New(errnie.WithError(err))
	}
	defer resp.Body.Close()

	if err = json.NewDecoder(resp.Body).Decode(&updatedLabel); err != nil {
		return Label{}, errnie.New(errnie.WithError(err))
	}

	return updatedLabel, nil
}

func (l *Labels) DeleteLabel(labelID int) (err error) {
	resp, err := l.client.doRequest(
		http.MethodDelete,
		fmt.Sprintf("/labels/%d", labelID),
		nil,
	)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}
	defer resp.Body.Close()

	return nil
}
