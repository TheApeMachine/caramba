package process

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/charmbracelet/log"
	"github.com/invopop/jsonschema"
	"github.com/spf13/viper"
	"github.com/theapemachine/errnie"
)

type Labelling struct {
	TicketID int   `json:"ticket_id" jsonschema:"required,description=The id of the ticket to label"`
	LabelIDs []int `json:"label_ids" jsonschema:"required,description=The ids of the labels to apply to the ticket"`
}

func NewLabelling() *Labelling {
	log.Info("NewLabelling")
	return &Labelling{}
}

func (labelling *Labelling) SystemPrompt(key string) string {
	log.Info("SystemPrompt", "key", key)
	labels := labelling.ListLabels()

	formattedLabels := []string{}
	for _, label := range labels.Data {
		formattedLabels = append(
			formattedLabels,
			fmt.Sprintf("- %d: %s", label.ID, label.Name),
		)
	}

	prompt := viper.GetViper().GetString(fmt.Sprintf("ai.setups.%s.processes.trengo.prompt", key))
	prompt = strings.ReplaceAll(prompt, "{{labels}}", strings.Join(formattedLabels, "\n"))
	prompt = strings.ReplaceAll(prompt, "{{schemas}}", labelling.GenerateSchema())

	return prompt
}

func (labelling *Labelling) GenerateSchema() string {
	log.Info("GenerateSchema")
	schema := jsonschema.Reflect(&Labelling{})
	out, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		errnie.Error(err)
	}

	return string(out)
}

func (labelling *Labelling) ListLabels() Labels {
	log.Info("ListLabels")
	url := "https://app.trengo.com/api/v2/labels"

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		errnie.Error(err)
	}

	req.Header.Add("Authorization", fmt.Sprintf(
		"Bearer %s", os.Getenv("TRENGO_API_TOKEN"),
	))
	req.Header.Add("accept", "application/json")
	req.Header.Add("content-type", "application/json")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		errnie.Error(err)
	}

	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		errnie.Error(err)
	}

	var labels Labels

	if err := json.Unmarshal(body, &labels); err != nil {
		errnie.Error(err)
	}

	return labels
}

type Labels struct {
	Data []struct {
		ID        int    `json:"id"`
		Name      string `json:"name"`
		Slug      string `json:"slug"`
		Color     string `json:"color"`
		SortOrder int    `json:"sort_order"`
		Archived  any    `json:"archived"`
	} `json:"data"`
	Links struct {
		First string `json:"first"`
		Last  string `json:"last"`
		Prev  any    `json:"prev"`
		Next  any    `json:"next"`
	} `json:"links"`
	Meta struct {
		CurrentPage int `json:"current_page"`
		From        int `json:"from"`
		LastPage    int `json:"last_page"`
		Links       []struct {
			URL    any    `json:"url"`
			Label  string `json:"label"`
			Active bool   `json:"active"`
		} `json:"links"`
		Path    string `json:"path"`
		PerPage int    `json:"per_page"`
		To      int    `json:"to"`
		Total   int    `json:"total"`
	} `json:"meta"`
}
