package sampling

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// DefaultManager is the default implementation of SamplingManager
type DefaultManager struct {
	defaultPrefs ModelPreferences
	mu           sync.RWMutex
}

// NewDefaultManager creates a new DefaultManager instance
func NewDefaultManager() *DefaultManager {
	return &DefaultManager{
		defaultPrefs: ModelPreferences{
			Temperature:      0.7,
			MaxTokens:        2048,
			TopP:             1.0,
			FrequencyPenalty: 0.0,
			PresencePenalty:  0.0,
		},
	}
}

// isEmptyPreferences checks if model preferences are empty/default values
func isEmptyPreferences(prefs ModelPreferences) bool {
	return prefs.Temperature == 0 && prefs.MaxTokens == 0 && prefs.TopP == 0
}

// CreateMessage creates a new message using the provided options
func (m *DefaultManager) CreateMessage(ctx context.Context, content string, opts SamplingOptions) (*SamplingResult, error) {
	startTime := time.Now()

	// Apply default preferences if not specified
	if isEmptyPreferences(opts.ModelPreferences) {
		m.mu.RLock()
		opts.ModelPreferences = m.defaultPrefs
		m.mu.RUnlock()
	}

	// TODO: Implement actual model interaction here
	// For now, just echo back the content
	message := Message{
		ID:        uuid.New().String(),
		Role:      "assistant",
		Content:   content,
		CreatedAt: time.Now(),
	}

	// Simulate token usage
	usage := Usage{
		PromptTokens:     len(content) / 4, // Rough approximation
		CompletionTokens: len(content) / 4,
		TotalTokens:      len(content) / 2,
	}

	return &SamplingResult{
		Message:  message,
		Usage:    usage,
		Duration: time.Since(startTime).Seconds(),
	}, nil
}

// StreamMessage streams message tokens as they are generated
func (m *DefaultManager) StreamMessage(ctx context.Context, content string, opts SamplingOptions) (<-chan *SamplingResult, error) {
	resultChan := make(chan *SamplingResult)

	// Apply default preferences if not specified
	if isEmptyPreferences(opts.ModelPreferences) {
		m.mu.RLock()
		opts.ModelPreferences = m.defaultPrefs
		m.mu.RUnlock()
	}

	go func() {
		defer close(resultChan)

		startTime := time.Now()
		messageID := uuid.New().String()

		// TODO: Implement actual streaming model interaction here
		// For now, just stream back the content character by character
		for i := 0; i < len(content); i++ {
			select {
			case <-ctx.Done():
				return
			default:
				message := Message{
					ID:        messageID,
					Role:      "assistant",
					Content:   string(content[i]),
					CreatedAt: time.Now(),
				}

				usage := Usage{
					PromptTokens:     1,
					CompletionTokens: 1,
					TotalTokens:      2,
				}

				resultChan <- &SamplingResult{
					Message:  message,
					Usage:    usage,
					Duration: time.Since(startTime).Seconds(),
				}

				time.Sleep(50 * time.Millisecond) // Simulate processing time
			}
		}
	}()

	return resultChan, nil
}

// GetModelPreferences returns the default model preferences
func (m *DefaultManager) GetModelPreferences(ctx context.Context) (*ModelPreferences, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	prefs := m.defaultPrefs
	return &prefs, nil
}

// UpdateModelPreferences updates the default model preferences
func (m *DefaultManager) UpdateModelPreferences(ctx context.Context, prefs ModelPreferences) error {
	// Validate preferences
	if prefs.Temperature < 0 || prefs.Temperature > 2 {
		return fmt.Errorf("temperature must be between 0 and 2")
	}
	if prefs.TopP < 0 || prefs.TopP > 1 {
		return fmt.Errorf("topP must be between 0 and 1")
	}
	if prefs.MaxTokens < 1 {
		return fmt.Errorf("maxTokens must be greater than 0")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.defaultPrefs = prefs
	return nil
}
