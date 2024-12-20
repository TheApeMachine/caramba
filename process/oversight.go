package process

// import (
// 	"encoding/json"
// 	"fmt"
// 	"strings"

// 	"github.com/invopop/jsonschema"
// 	"github.com/spf13/viper"
// )

// type Oversight struct {
// 	Memory      *Memory
// 	Performance *Performance
// 	Integration *Integration
// }

// func NewOversight() *Oversight {
// 	return &Oversight{}
// }

// func (o *Oversight) SystemPrompt(key string) string {
// 	prompt := viper.GetViper().GetString(fmt.Sprintf("ai.setups.%s.processes.oversight.prompt", key))
// 	prompt = strings.ReplaceAll(prompt, "{{schemas}}", o.GenerateSchema())
// 	return prompt
// }

// func (o *Oversight) GenerateSchema() string {
// 	schema := jsonschema.Reflect(&Oversight{})
// 	out, err := json.MarshalIndent(schema, "", "  ")
// 	if err != nil {
// 		return ""
// 	}
// 	return string(out)
// }
