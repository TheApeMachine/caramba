package provider

import (
	"errors"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/theapemachine/errnie"
)

type ProviderStatus struct {
	name     string
	provider Provider
	occupied bool
	lastUsed time.Time
	failures int
	mu       sync.Mutex
}

type BalancedProvider struct {
	providers   []*ProviderStatus
	selectIndex int
	initMu      sync.Mutex
	initialized bool
}

var (
	balancedProviderInstance *BalancedProvider
	onceBalancedProvider     sync.Once
)

func NewBalancedProvider() *BalancedProvider {
	onceBalancedProvider.Do(func() {
		// Initialize providers
		providers := make([]*ProviderStatus, 0)

		// OpenAI provider
		if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey != "" {
			openaiProvider := NewOpenAI(openaiKey, openai.ChatModelGPT4oMini)
			if openaiProvider != nil {
				providers = append(providers, &ProviderStatus{
					name:     "gpt-4o-mini",
					provider: openaiProvider,
					occupied: false,
				})
			}
		}

		// Anthropic provider
		if anthropicKey := os.Getenv("ANTHROPIC_API_KEY"); anthropicKey != "" {
			anthropicProvider := NewAnthropic(anthropicKey, anthropic.ModelClaude3_5Sonnet20241022)
			if anthropicProvider != nil {
				providers = append(providers, &ProviderStatus{
					name:     "claude-3-5-sonnet",
					provider: anthropicProvider,
					occupied: false,
				})
			}
		}

		// Gemini provider
		if geminiKey := os.Getenv("GEMINI_API_KEY"); geminiKey != "" {
			geminiProvider := NewGoogle(geminiKey, "gemini-1.5-flash")
			if geminiProvider != nil {
				providers = append(providers, &ProviderStatus{
					name:     "gemini-1.5-flash",
					provider: geminiProvider,
					occupied: false,
				})
			}
		}

		// Cohere provider
		if cohereKey := os.Getenv("COHERE_API_KEY"); cohereKey != "" {
			cohereProvider := NewCohere(cohereKey, "command-r")
			if cohereProvider != nil {
				providers = append(providers, &ProviderStatus{
					name:     "command-r",
					provider: cohereProvider,
					occupied: false,
				})
			}
		}

		balancedProviderInstance = &BalancedProvider{
			providers:   providers,
			selectIndex: 0,
			initialized: false,
		}

		if len(providers) == 0 {
			errnie.Error(errors.New("no valid providers initialized"))
		}
	})

	return balancedProviderInstance
}

func (lb *BalancedProvider) Generate(params GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		provider := lb.getAvailableProvider()
		if provider == nil || provider.provider == nil {
			out <- Event{
				Type:  EventError,
				Error: errors.New("no available provider found or provider is nil"),
			}
			return
		}

		provider.mu.Lock()
		provider.occupied = true
		provider.lastUsed = time.Now()
		provider.mu.Unlock()

		for event := range provider.provider.Generate(params) {
			out <- event
		}

		provider.mu.Lock()
		provider.occupied = false
		provider.mu.Unlock()
	}()

	return out
}

func (lb *BalancedProvider) getAvailableProvider() *ProviderStatus {
	if provider := lb.handleFirstRequest(); provider != nil {
		return provider
	}
	return lb.findBestAvailableProvider()
}

func (lb *BalancedProvider) handleFirstRequest() *ProviderStatus {
	lb.initMu.Lock()
	defer lb.initMu.Unlock()

	if lb.initialized {
		return nil
	}

	availableProviders := lb.getUnoccupiedProviders()
	if len(availableProviders) == 0 {
		return nil
	}

	selected := availableProviders[rand.Intn(len(availableProviders))]
	lb.markProviderAsOccupied(selected)
	lb.initialized = true

	return selected
}

func (lb *BalancedProvider) findBestAvailableProvider() *ProviderStatus {
	maxAttempts := 10
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if provider := lb.selectBestProvider(); provider != nil {
			return provider
		}
		errnie.Warn("all providers occupied or in cooldown, attempt %d, waiting...", attempt+1)
		time.Sleep(1 * time.Second)
	}

	errnie.Error(errors.New("no providers available after maximum attempts"))
	return nil
}

func (lb *BalancedProvider) selectBestProvider() *ProviderStatus {
	var bestProvider *ProviderStatus
	oldestUse := time.Now()
	cooldownPeriod := 60 * time.Second
	maxFailures := 3

	for _, ps := range lb.providers {
		ps.mu.Lock()

		if !lb.isProviderAvailable(ps, cooldownPeriod, maxFailures) {
			ps.mu.Unlock()
			continue
		}

		if lb.isBetterProvider(ps, bestProvider, oldestUse) {
			bestProvider = ps
			oldestUse = ps.lastUsed
		}
		ps.mu.Unlock()
	}

	if bestProvider != nil {
		lb.markProviderAsOccupied(bestProvider)
	}

	return bestProvider
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

func (lb *BalancedProvider) markProviderAsOccupied(ps *ProviderStatus) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.occupied = true
	ps.lastUsed = time.Now()
}
