package provider

type LLMGenerationParams struct {
	Thread           *Thread
	Tools            []Tool
	Process          Process
	Temperature      float64
	MaxTokens        int64
	TopP             float64
	TopK             int64
	FrequencyPenalty float64
	PresencePenalty  float64
}

func NewGenerationParams(thread *Thread) *LLMGenerationParams {
	return &LLMGenerationParams{
		Thread:           thread,
		Tools:            make([]Tool, 0),
		Process:          nil,
		Temperature:      0.5,
		MaxTokens:        4096,
		TopP:             0.9,
		TopK:             50,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
	}
}
