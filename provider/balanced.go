package provider

import (
	"context"
	"errors"
	"os"
	"sync"
	"time"

	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
ProviderStatus wraps the current status of a provider, which is used
in the balancing process.
*/
type ProviderStatus struct {
	provider Provider
	occupied bool
	lastUsed time.Time
	failures int
	mu       sync.Mutex
}

/*
BalancedProvider wraps multiple LLM providers and balances generation
requests across them. The main reason behind this is to avoid rate-limiting,
single-point-of-failure, and single-provider bias.
*/
type BalancedProvider struct {
	*BaseProvider
	providers   []*ProviderStatus
	selectIndex int
	initMu      sync.Mutex
	initialized bool
	cancel      context.CancelFunc
}

var (
	balancedProviderInstance *BalancedProvider
	onceBalancedProvider     sync.Once
)

/*
NewBalancedProvider returns a provider that agents can use, which makes
the balancing process transparent to the user.
*/
func NewBalancedProvider() *BalancedProvider {
	return &BalancedProvider{
		BaseProvider: NewBaseProvider(),
		providers:    make([]*ProviderStatus, 0),
		selectIndex:  0,
	}
}

/*
Name returns the name of the provider.
*/
func (lb *BalancedProvider) Name() string {
	return "balanced"
}

/*
Generate a response from an LLM provider, passing in the parameters, which
include the messages, and model settings to use.
*/
func (lb *BalancedProvider) Generate(ctx context.Context, params *LLMGenerationParams) <-chan Event {
	hasSystem := false
	hasUser := false

	for _, message := range params.Thread.Messages {
		if message.Role == "system" {
			hasSystem = true
		} else if message.Role == "user" {
			hasUser = true
		}
	}

	if !hasSystem {
		errnie.Error(errors.New("no system message found"))
		return nil
	}

	if !hasUser {
		errnie.Warn("no user message found")
	}

	out := make(chan Event)
	ctx, cancel := context.WithCancel(ctx)
	lb.cancel = cancel

	go func() {
		defer close(out)
		defer cancel()

		// Send start event
		startEvent := NewEventData()
		startEvent.EventType = EventStart
		startEvent.Name = "balanced_generation_start"
		out <- startEvent

		provider := lb.getAvailableProvider()
		if provider == nil || provider.provider == nil {
			errEvent := NewEventData()
			errEvent.EventType = EventError
			errEvent.Error = errors.New("no available provider found")
			errEvent.Name = "balanced_error"
			out <- errEvent
			return
		}

		provider.mu.Lock()
		provider.occupied = true
		provider.lastUsed = time.Now()
		provider.mu.Unlock()

		errnie.Info("generating response from %s", provider.provider.Name())
		events := provider.provider.Generate(ctx, params)

		if events == nil {
			errnie.Error(errors.New("events channel is nil"))
			return
		}

		for event := range events {
			select {
			case <-ctx.Done():
				return
			case out <- event:
			}
		}

		provider.mu.Lock()
		provider.occupied = false
		provider.mu.Unlock()

		// Send done event
		doneEvent := NewEventData()
		doneEvent.EventType = EventDone
		doneEvent.Name = "balanced_generation_complete"
		out <- doneEvent
	}()

	return out
}

/*
CancelGeneration cancels any ongoing generation.
*/
func (lb *BalancedProvider) CancelGeneration(ctx context.Context) error {
	if lb.cancel != nil {
		lb.cancel()
	}
	return nil
}

func (lb *BalancedProvider) getAvailableProvider() *ProviderStatus {
	// Try to handle first request if not initialized
	if !lb.initialized {
		if provider := lb.handleFirstRequest(); provider != nil {
			return provider
		}
	}

	cooldownPeriod := 60 * time.Second
	maxFailures := 3
	maxAttempts := 10

	for attempt := 0; attempt < maxAttempts; attempt++ {
		var bestProvider *ProviderStatus
		oldestUse := time.Now()

		// First try to find a provider that's never been used
		for _, ps := range lb.providers {
			ps.mu.Lock()
			if !ps.occupied && ps.provider != nil && ps.lastUsed.IsZero() {
				ps.occupied = true
				ps.lastUsed = time.Now()
				ps.mu.Unlock()
				return ps
			}
			ps.mu.Unlock()
		}

		// If no unused providers, try to find the best available one
		for _, ps := range lb.providers {
			ps.mu.Lock()
			if !ps.occupied && ps.provider != nil &&
				(ps.failures < maxFailures || time.Since(ps.lastUsed) >= cooldownPeriod) &&
				(bestProvider == nil || ps.failures < bestProvider.failures ||
					(ps.failures == bestProvider.failures && ps.lastUsed.Before(oldestUse))) {
				bestProvider = ps
				oldestUse = ps.lastUsed
			}
			ps.mu.Unlock()
		}

		if bestProvider != nil {
			bestProvider.mu.Lock()
			bestProvider.occupied = true
			bestProvider.lastUsed = time.Now()
			bestProvider.mu.Unlock()
			return bestProvider
		}

		errnie.Warn("all providers occupied or in cooldown, attempt %d, waiting...", attempt+1)
		time.Sleep(time.Second)
	}

	errnie.Error(errors.New("no providers available after maximum attempts"))
	return nil
}

func (lb *BalancedProvider) handleFirstRequest() *ProviderStatus {
	lb.initMu.Lock()
	defer lb.initMu.Unlock()

	// If already initialized, let the normal selection process handle it
	if lb.initialized {
		return nil
	}

	// Initialize providers if not done yet
	if len(lb.providers) == 0 {
		if err := lb.InitializeProviders(); err != nil {
			errnie.Error(err)
			return nil
		}
	}

	// Find an available provider
	for _, ps := range lb.providers {
		ps.mu.Lock()
		if !ps.occupied && ps.provider != nil {
			ps.occupied = true
			ps.lastUsed = time.Now()
			ps.mu.Unlock()
			lb.initialized = true
			return ps
		}
		ps.mu.Unlock()
	}

	// If we get here, no providers were available
	return nil
}

func (lb *BalancedProvider) isProviderAvailable(ps *ProviderStatus, cooldownPeriod time.Duration, maxFailures int) bool {
	if ps.occupied || ps.provider == nil {
		return false
	}

	if ps.failures >= maxFailures && time.Since(ps.lastUsed) < cooldownPeriod {
		return false
	}

	if ps.failures >= maxFailures && time.Since(ps.lastUsed) >= cooldownPeriod {
		ps.failures = 0
	}

	return true
}

func (lb *BalancedProvider) isBetterProvider(candidate, current *ProviderStatus, oldestUse time.Time) bool {
	return current == nil ||
		candidate.failures < current.failures ||
		(candidate.failures == current.failures && candidate.lastUsed.Before(oldestUse))
}

func (lb *BalancedProvider) getUnoccupiedProviders() []*ProviderStatus {
	available := make([]*ProviderStatus, 0)
	for _, ps := range lb.providers {
		ps.mu.Lock()
		if !ps.occupied {
			available = append(available, ps)
		}
		ps.mu.Unlock()
	}
	return available
}

func (lb *BalancedProvider) selectBestProvider() *ProviderStatus {
	var bestProvider *ProviderStatus
	oldestUse := time.Now()
	cooldownPeriod := 60 * time.Second
	maxFailures := 3

	for _, ps := range lb.providers {
		ps.mu.Lock()
		if lb.isProviderAvailable(ps, cooldownPeriod, maxFailures) &&
			lb.isBetterProvider(ps, bestProvider, oldestUse) {
			bestProvider = ps
			oldestUse = ps.lastUsed
		}
		ps.mu.Unlock()
	}

	if bestProvider != nil {
		bestProvider.mu.Lock()
		bestProvider.occupied = true
		bestProvider.lastUsed = time.Now()
		bestProvider.mu.Unlock()
	}

	return bestProvider
}

// Cleanup performs any necessary cleanup
func (bp *BalancedProvider) Cleanup(ctx context.Context) error {
	for _, p := range bp.providers {
		if err := p.provider.Cleanup(ctx); err != nil {
			return err
		}
	}
	return nil
}

// Configure sets up the provider with the given configuration
func (bp *BalancedProvider) Configure(config *ProviderConfig) error {
	for _, p := range bp.providers {
		if err := p.provider.Configure(config); err != nil {
			return err
		}
	}
	return nil
}

// GetCapabilities returns the provider capabilities
func (bp *BalancedProvider) GetCapabilities() map[string]interface{} {
	return map[string]interface{}{
		"balanced":  true,
		"providers": len(bp.providers),
		"streaming": true,
		"tools":     true,
	}
}

// GetConfig returns the current provider configuration
func (bp *BalancedProvider) GetConfig() *ProviderConfig {
	// Return config from first available provider since they all share the same config
	for _, p := range bp.providers {
		if p.provider != nil {
			return p.provider.GetConfig()
		}
	}
	return nil
}

// GetMetrics returns the provider metrics
func (bp *BalancedProvider) GetMetrics() (*ProviderMetrics, error) {
	metrics := &ProviderMetrics{
		CustomMetrics: make(map[string]interface{}),
	}

	for i, p := range bp.providers {
		if p.provider != nil {
			metrics.CustomMetrics[p.provider.Name()] = map[string]interface{}{
				"failures": p.failures,
				"occupied": p.occupied,
				"lastUsed": p.lastUsed,
				"index":    i,
			}
		}
	}
	return metrics, nil
}

// HealthCheck performs a health check on all providers
func (bp *BalancedProvider) HealthCheck(ctx context.Context) *utils.HealthStatus {
	status := utils.StatusHealthy
	for _, p := range bp.providers {
		if p.provider != nil && p.failures > 3 {
			status = utils.StatusDegraded
			break
		}
	}
	return &status
}

// Initialize sets up the provider
func (bp *BalancedProvider) Initialize(ctx context.Context) error {
	bp.initMu.Lock()
	defer bp.initMu.Unlock()

	// Initialize providers if not done yet
	if len(bp.providers) == 0 {
		if err := bp.InitializeProviders(); err != nil {
			return err
		}
	}

	// Initialize each provider
	for _, p := range bp.providers {
		if err := p.provider.Initialize(ctx); err != nil {
			return err
		}
		// Ensure provider starts in unoccupied state
		p.mu.Lock()
		p.occupied = false
		p.lastUsed = time.Time{}
		p.failures = 0
		p.mu.Unlock()
	}

	bp.initialized = true
	return nil
}

// PauseGeneration pauses generation for all providers
func (bp *BalancedProvider) PauseGeneration() error {
	for _, p := range bp.providers {
		if err := p.provider.PauseGeneration(); err != nil {
			return err
		}
	}
	return nil
}

// ResumeGeneration resumes generation for all providers
func (bp *BalancedProvider) ResumeGeneration() error {
	for _, p := range bp.providers {
		if err := p.provider.ResumeGeneration(); err != nil {
			return err
		}
	}
	return nil
}

// SupportsFeature checks if a feature is supported
func (bp *BalancedProvider) SupportsFeature(feature string) bool {
	return feature == "balanced"
}

// ValidateConfig validates the configuration for all providers
func (bp *BalancedProvider) ValidateConfig() error {
	for _, p := range bp.providers {
		if err := p.provider.ValidateConfig(); err != nil {
			return err
		}
	}
	return nil
}

// Version returns the provider version
func (bp *BalancedProvider) Version() string {
	return "1.0.0"
}

// AddProvider adds a new provider to the balanced provider
func (bp *BalancedProvider) AddProvider(provider Provider) {
	bp.providers = append(bp.providers, &ProviderStatus{
		provider: provider,
		occupied: false,
		lastUsed: time.Time{},
		failures: 0,
	})
}

// InitializeProviders initializes all configured providers based on environment variables
func (bp *BalancedProvider) InitializeProviders() error {
	// OpenAI
	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey != "" {
		bp.AddProvider(NewOpenAI(openaiKey))
	}

	// Anthropic
	if anthropicKey := os.Getenv("ANTHROPIC_API_KEY"); anthropicKey != "" {
		bp.AddProvider(NewAnthropic(anthropicKey))
	}

	// Gemini
	if geminiKey := os.Getenv("GEMINI_API_KEY"); geminiKey != "" {
		bp.AddProvider(NewGemini(geminiKey))
	}

	// Cohere
	if cohereKey := os.Getenv("COHERE_API_KEY"); cohereKey != "" {
		bp.AddProvider(NewCohere(cohereKey))
	}

	// Check if any providers were added
	if len(bp.providers) == 0 {
		return errors.New("no providers were initialized")
	}

	return nil
}
