package chat

import (
	"fmt"
	"strings"
)

const promptTemplatePlaceholder = "{{prompt}}"

/*
promptTemplate formats user text before tokenizer encoding.
*/
type promptTemplate struct {
	text string
}

/*
newPromptTemplate validates a manifest-owned prompt template.
*/
func newPromptTemplate(text string) (promptTemplate, error) {
	if strings.TrimSpace(text) == "" {
		return promptTemplate{}, nil
	}

	if !strings.Contains(text, promptTemplatePlaceholder) {
		return promptTemplate{}, fmt.Errorf(
			"chat.model: prompt_template must contain %s",
			promptTemplatePlaceholder,
		)
	}

	return promptTemplate{text: text}, nil
}

/*
Apply returns the model prompt text.
*/
func (template promptTemplate) Apply(prompt string) string {
	if template.text == "" {
		return prompt
	}

	return strings.ReplaceAll(template.text, promptTemplatePlaceholder, prompt)
}
