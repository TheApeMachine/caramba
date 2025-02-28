/*
Package llm provides integrations with various Language Model providers.
This package implements the core.LLMProvider interface for different providers
like Anthropic, OpenAI, and others, as well as utility providers like BalancedProvider.
*/
package llm

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/hub"
)

/*
ProviderStatus tracks the operational status of a provider.
It maintains information about availability, errors, and cooldown periods
for individual LLM providers within the BalancedProvider.
*/
type ProviderStatus struct {
	/* Provider is the underlying LLM provider implementation */
	Provider core.LLMProvider
	/* Available indicates whether the provider is currently operational */
	Available bool
	/* LastError stores the most recent error encountered with this provider */
	LastError error
	/* ErrorCount tracks how many consecutive errors have occurred */
	ErrorCount int
	/* LastUsed records when the provider was last used */
	LastUsed time.Time
	/* CooldownTime is the duration to wait after errors before trying again */
	CooldownTime time.Duration
}

/*
BalancedProvider implements the LLMProvider interface by load balancing across multiple providers.
It automatically routes requests to available providers, handles error cases,
implements cooldown periods for failing providers, and offers fault tolerance.
*/
type BalancedProvider struct {
	hub *hub.Queue
	/* providers holds the status of all underlying LLM providers */
	providers []*ProviderStatus
	/* mu is a mutex for thread-safe access to provider statuses */
	mu sync.RWMutex
	/* roundRobinIdx tracks the next provider to use in round-robin allocation */
	roundRobinIdx int
	/* maxRetries is the number of attempts to make before giving up */
	maxRetries int
	/* errorThreshold is the number of errors before a provider is temporarily disabled */
	errorThreshold int
}

/*
NewBalancedProvider creates a new load-balancing provider.
It initializes the BalancedProvider with a collection of LLM providers
and sets default values for retries and error thresholds.

Parameters:
  - providers: A slice of LLM providers to balance between

Returns:
  - A pointer to the initialized BalancedProvider
*/
func NewBalancedProvider(providers []core.LLMProvider) *BalancedProvider {
	bp := &BalancedProvider{
		hub:            hub.NewQueue(),
		providers:      make([]*ProviderStatus, 0, len(providers)),
		maxRetries:     3,
		errorThreshold: 5,
	}

	for _, p := range providers {
		bp.providers = append(bp.providers, &ProviderStatus{
			Provider:     p,
			Available:    true,
			CooldownTime: 10 * time.Second,
		})
	}

	return bp
}

/*
Name returns the name of the provider.
This is used for identification and logging purposes.

Returns:
  - The string "balanced"
*/
func (p *BalancedProvider) Name() string {
	return "balanced"
}

/*
GenerateResponse generates a response by selecting from available providers.
It automatically tries different providers if some fail, implementing
a fault-tolerant approach to generation.

Parameters:
  - ctx: The context for the request, which can be used for cancellation
  - prompt: The user input to send to the model
  - options: Configuration options for the generation process

Returns:
  - The generated text response from one of the successful providers
  - An error if all providers fail after the maximum retry attempts
*/
func (p *BalancedProvider) GenerateResponse(
	ctx context.Context,
	params core.LLMParams,
) core.LLMResponse {
	if len(p.providers) == 0 {
		return core.LLMResponse{
			Error: errors.New("no providers configured"),
		}
	}

	// Try to get a response for up to maxRetries attempts
	var lastError error

	for range p.maxRetries {
		provider, err := p.getNextAvailableProvider()

		if err != nil {
			p.hub.Add(hub.NewEvent(
				p.Name(),
				"llm",
				"error",
				hub.EventTypeError,
				err.Error(),
				map[string]string{},
			))
		}

		resp := provider.Provider.GenerateResponse(ctx, params)
		p.updateProviderStatus(provider, resp.Error)

		if resp.Error == nil {
			return resp
		}

		lastError = resp.Error
		p.hub.Add(hub.NewEvent(
			p.Name(),
			"llm",
			"error",
			hub.EventTypeError,
			resp.Error.Error(),
			map[string]string{},
		))
	}

	return core.LLMResponse{
		Error: errors.New("all LLM providers failed: " + lastError.Error()),
	}
}

/*
StreamResponse streams a response from a selected provider.
It attempts to stream from available providers, retrying with different
providers if some fail, implementing a fault-tolerant approach.

Parameters:
  - ctx: The context for the request, which can be used for cancellation
  - prompt: The user input to send to the model
  - options: Configuration options for the generation process
  - handler: A callback function that receives each text chunk as it's generated

Returns:
  - An error if all providers fail after the maximum retry attempts
*/
func (p *BalancedProvider) StreamResponse(
	ctx context.Context,
	params core.LLMParams,
) <-chan core.LLMResponse {
	if len(p.providers) == 0 {
		return nil
	}

	out := make(chan core.LLMResponse)

	go func() {
		defer close(out)

		for range p.maxRetries {
			provider, err := p.getNextAvailableProvider()

			if err != nil {
				p.hub.Add(hub.NewEvent(
					p.Name(),
					"llm",
					"error",
					hub.EventTypeError,
					err.Error(),
					map[string]string{},
				))
			}

			stream := provider.Provider.StreamResponse(ctx, params)
			p.updateProviderStatus(provider, err)

			for chunk := range stream {
				out <- chunk
			}
		}
	}()

	return out
}

/*
getNextAvailableProvider selects the next available provider.
It implements a round-robin selection strategy for load balancing,
skipping providers that are in a cooldown period due to errors.

Returns:
  - A pointer to the selected provider's status
  - An error if no providers are available
*/
func (p *BalancedProvider) getNextAvailableProvider() (*ProviderStatus, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Check if we have any available providers
	startIdx := p.roundRobinIdx
	for i := 0; i < len(p.providers); i++ {
		idx := (startIdx + i) % len(p.providers)
		provider := p.providers[idx]

		// Reset provider status if cooldown period has passed
		if !provider.Available && time.Since(provider.LastUsed) > provider.CooldownTime {
			provider.Available = true
			provider.ErrorCount = 0
		}

		if provider.Available {
			p.roundRobinIdx = (idx + 1) % len(p.providers)
			return provider, nil
		}
	}

	return nil, errors.New("no available LLM providers")
}

/*
updateProviderStatus updates the status of a provider after use.
It tracks errors, updates availability status, and manages
cooldown periods for providers experiencing errors.

Parameters:
  - provider: The provider status to update
  - err: The error encountered, or nil if the operation was successful
*/
func (p *BalancedProvider) updateProviderStatus(provider *ProviderStatus, err error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	provider.LastUsed = time.Now()

	if err != nil {
		provider.LastError = err
		provider.ErrorCount++

		// If error count exceeds threshold, mark as unavailable
		if provider.ErrorCount >= p.errorThreshold {
			provider.Available = false
			// Increase cooldown time exponentially after repeated errors
			provider.CooldownTime = provider.CooldownTime * 2
			if provider.CooldownTime > 5*time.Minute {
				provider.CooldownTime = 5 * time.Minute
			}
		}
	} else {
		// Gradually reduce error count on successful calls
		if provider.ErrorCount > 0 {
			provider.ErrorCount--
		}
		// Reset cooldown time on successful calls
		provider.CooldownTime = 10 * time.Second
	}
}

/*
AddProvider adds a new provider to the balancer.
This allows for dynamically adding new providers at runtime.

Parameters:
  - provider: The LLM provider to add to the balancer
*/
func (p *BalancedProvider) AddProvider(provider core.LLMProvider) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.providers = append(p.providers, &ProviderStatus{
		Provider:     provider,
		Available:    true,
		CooldownTime: 10 * time.Second,
	})
}

/*
RemoveProvider removes a provider by name.
This allows for dynamically removing providers at runtime.

Parameters:
  - name: The name of the provider to remove

Returns:
  - true if the provider was found and removed, false otherwise
*/
func (p *BalancedProvider) RemoveProvider(name string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()

	for i, provider := range p.providers {
		if provider.Provider.Name() == name {
			p.providers = append(p.providers[:i], p.providers[i+1:]...)
			return true
		}
	}
	return false
}

/*
GetProviderStatuses returns the current status of all providers.
This is useful for monitoring and debugging the balanced provider.

Returns:
  - A slice of maps containing status information for each provider
*/
func (p *BalancedProvider) GetProviderStatuses() []map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	statuses := make([]map[string]interface{}, len(p.providers))
	for i, provider := range p.providers {
		var errStr string
		if provider.LastError != nil {
			errStr = provider.LastError.Error()
		}

		statuses[i] = map[string]interface{}{
			"name":          provider.Provider.Name(),
			"available":     provider.Available,
			"error_count":   provider.ErrorCount,
			"last_error":    errStr,
			"last_used":     provider.LastUsed,
			"cooldown_time": provider.CooldownTime.Seconds(),
		}
	}

	return statuses
}
