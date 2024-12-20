package process

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/spf13/viper"
)

type Performance struct {
	Metrics []Metric `json:"metrics" jsonschema:"required,title=Metrics,description=Metrics to monitor"`
}

type Metric struct {
	Name        string `json:"name" jsonschema:"required,title=Name,description=The name of the metric."`
	Description string `json:"description" jsonschema:"required,title=Description,description=The description of the metric."`
	Threshold   string `json:"threshold" jsonschema:"required,title=Threshold,description=The threshold of the metric."`
	Operator    string `json:"operator" jsonschema:"required,title=Operator,description=The operator of the metric."`
	Value       string `json:"value" jsonschema:"required,title=Value,description=The value of the metric."`
}

func NewPerformance() *Performance {
	return &Performance{}
}

func (p *Performance) SystemPrompt(key string) string {
	prompt := viper.GetViper().GetString(fmt.Sprintf("ai.setups.%s.processes.performance.prompt", key))
	prompt = strings.ReplaceAll(prompt, "{{schemas}}", p.GenerateSchema())
	return prompt
}

func (p *Performance) GenerateSchema() string {
	schema := jsonschema.Reflect(&Performance{})
	out, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return ""
	}
	return string(out)
}
