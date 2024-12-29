package provider

type GenerationParams struct {
	Thread      *Thread
	Tools       []Tool
	Process     Process
	Temperature float64
	MaxTokens   int64
}

func NewGenerationParams() *GenerationParams {
	return &GenerationParams{
		Thread:      NewThread(),
		Tools:       make([]Tool, 0),
		Process:     nil,
		Temperature: 0.5,
		MaxTokens:   4096,
	}
}
