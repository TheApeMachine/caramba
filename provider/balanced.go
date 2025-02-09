package provider

import (
	"bytes"
	"context"
	"errors"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
ProviderStatus wraps a provider with its health status.
*/
type ProviderStatus struct {
	provider      Provider
	cooldownUntil time.Time
	errorCount    int
}

/*
BalancedProvider wraps multiple LLM providers and balances generation
requests across them, with automatic failover and cooldown on errors.
*/
type BalancedProvider struct {
	*BaseProvider
	mu        sync.RWMutex
	providers []*ProviderStatus
	rwBuffer  *bytes.Buffer // Buffer for IO operations
}

/*
NewBalancedProvider creates and returns a new BalancedProvider instance.
*/
func NewBalancedProvider() *BalancedProvider {
	bp := &BalancedProvider{
		BaseProvider: NewBaseProvider(),
		providers:    make([]*ProviderStatus, 0),
		rwBuffer:     new(bytes.Buffer),
	}
	bp.Initialize(context.Background())
	return bp
}

/*
Name returns the identifier for this provider type.

Returns:

	string: The string "balanced" which identifies this provider type
*/
func (lb *BalancedProvider) Name() string {
	return "balanced"
}

/*
Generate produces a response from an available provider, with automatic
retry on failure.
*/
func (lb *BalancedProvider) Generate(params *LLMGenerationParams) <-chan *Event {
	out := make(chan *Event)
	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		defer close(out)
		defer cancel()

		// Try up to 3 times with different providers
		for attempt := 0; attempt < 3; attempt++ {
			provider := lb.getHealthyProvider()
			if provider == nil {
				out <- NewEvent("generate:error", EventError, "no healthy providers available", "", nil)
				return
			}

			events := provider.provider.Generate(params)
			if events == nil {
				lb.markProviderUnhealthy(provider, "nil events channel")
				continue
			}

			// Try this provider
			failed := false
			for event := range events {
				if event.Type == EventError {
					lb.markProviderUnhealthy(provider, event.Text)
					failed = true
					break
				}
				select {
				case <-ctx.Done():
					return
				case out <- event:
				}
			}

			// If we successfully processed all events, we're done
			if !failed {
				return
			}
		}

		// If we get here, we've exhausted our retries
		out <- NewEvent("generate:error", EventError, "all provider attempts failed", "", nil)
	}()

	return out
}

/*
getHealthyProvider returns a random healthy provider from the pool.
*/
func (lb *BalancedProvider) getHealthyProvider() *ProviderStatus {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	// Get all healthy providers
	var healthy []*ProviderStatus
	now := time.Now()
	for _, p := range lb.providers {
		if p.provider != nil && now.After(p.cooldownUntil) {
			healthy = append(healthy, p)
		}
	}

	if len(healthy) == 0 {
		return nil
	}

	// Randomly select one
	return healthy[rand.Intn(len(healthy))]
}

/*
markProviderUnhealthy marks a provider as unhealthy and puts it in cooldown.
*/
func (lb *BalancedProvider) markProviderUnhealthy(provider *ProviderStatus, reason string) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	provider.errorCount++
	cooldownDuration := time.Duration(provider.errorCount*5) * time.Second
	provider.cooldownUntil = time.Now().Add(cooldownDuration)

	errnie.Warn("provider marked unhealthy",
		"reason", reason,
		"cooldown", cooldownDuration,
		"errors", provider.errorCount,
	)
}

/*
CancelGeneration stops any ongoing generation process.
It cancels the context used in the Generate method, which will stop
the generation and close the event channel.

Parameters:

	ctx: Context for the cancellation operation (currently unused)

Returns:

	error: Any error that occurred during cancellation, nil if successful
*/
func (lb *BalancedProvider) CancelGeneration(ctx context.Context) error {
	return nil
}

/*
Cleanup performs necessary cleanup operations on all managed providers.
It should be called when the BalancedProvider is being shut down.

Parameters:

	ctx: Context for the cleanup operation

Returns:

	error: Any error that occurred during cleanup, nil if successful
*/
func (bp *BalancedProvider) Cleanup(ctx context.Context) error {
	for _, p := range bp.providers {
		if err := p.provider.Cleanup(ctx); err != nil {
			return err
		}
	}
	return nil
}

/*
Configure applies the provided configuration to all managed providers.
This ensures consistent configuration across all providers in the pool.

Parameters:

	config: The provider configuration to apply

Returns:

	error: Any error that occurred during configuration, nil if successful
*/
func (bp *BalancedProvider) Configure(config *ProviderConfig) error {
	for _, p := range bp.providers {
		if err := p.provider.Configure(config); err != nil {
			return err
		}
	}
	return nil
}

/*
GetCapabilities returns a map of the provider's capabilities.
This includes information about balancing, number of providers,
streaming support, and tool availability.

Returns:

	map[string]interface{}: A map describing the provider's capabilities
*/
func (bp *BalancedProvider) GetCapabilities() map[string]interface{} {
	return map[string]interface{}{
		"balanced":  true,
		"providers": len(bp.providers),
		"streaming": true,
		"tools":     true,
	}
}

/*
GetConfig retrieves the current configuration from the first available provider.
Since all providers share the same configuration, any provider's config is representative.

Returns:

	*ProviderConfig: The current provider configuration, or nil if no providers are available
*/
func (bp *BalancedProvider) GetConfig() *ProviderConfig {
	// Return config from first available provider since they all share the same config
	for _, p := range bp.providers {
		if p.provider != nil {
			return p.provider.GetConfig()
		}
	}
	return nil
}

/*
GetMetrics returns provider-specific metrics and statistics.
It collects metrics from all managed providers, including their failure counts,
occupied status, and last used timestamps.

Returns:

	*ProviderMetrics: The collected metrics for all providers
	error: Any error that occurred while collecting metrics
*/
func (bp *BalancedProvider) GetMetrics() (*ProviderMetrics, error) {
	metrics := &ProviderMetrics{
		CustomMetrics: make(map[string]interface{}),
	}

	for i, p := range bp.providers {
		if p.provider != nil {
			metrics.CustomMetrics[p.provider.Name()] = map[string]interface{}{
				"failures": p.errorCount,
				"cooldown": p.cooldownUntil,
				"index":    i,
			}
		}
	}
	return metrics, nil
}

/*
HealthCheck performs a health check on all managed providers.
It evaluates the health status based on provider failure counts,
marking the service as degraded if any provider has more than
3 failures.

Parameters:

	ctx: Context for the health check operation

Returns:

	*utils.HealthStatus: A pointer to either StatusHealthy or StatusDegraded
*/
func (bp *BalancedProvider) HealthCheck(ctx context.Context) *utils.HealthStatus {
	status := utils.StatusHealthy
	for _, p := range bp.providers {
		if p.provider != nil && p.errorCount > 3 {
			status = utils.StatusDegraded
			break
		}
	}
	return &status
}

/*
Initialize sets up the BalancedProvider and its managed providers.
It ensures thread-safe initialization of the provider pool, initializes
each provider, and resets their states to a clean initial condition.
This method must be called before the BalancedProvider can be used.

Parameters:

	ctx: Context for the initialization operation

Returns:

	error: Any error that occurred during initialization
*/
func (bp *BalancedProvider) Initialize(ctx context.Context) error {
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
	}

	return nil
}

/*
PauseGeneration temporarily stops generation across all managed providers.
It attempts to pause each provider in the pool, returning an error if any
provider fails to pause.

Returns:

	error: Any error that occurred while pausing providers
*/
func (bp *BalancedProvider) PauseGeneration() error {
	for _, p := range bp.providers {
		if err := p.provider.PauseGeneration(); err != nil {
			return err
		}
	}
	return nil
}

/*
ResumeGeneration restarts generation across all managed providers.
It attempts to resume each provider in the pool, returning an error if any
provider fails to resume.

Returns:

	error: Any error that occurred while resuming providers
*/
func (bp *BalancedProvider) ResumeGeneration() error {
	for _, p := range bp.providers {
		if err := p.provider.ResumeGeneration(); err != nil {
			return err
		}
	}
	return nil
}

/*
SupportsFeature checks if a specific feature is supported by this provider.
Currently only returns true for the "balanced" feature, indicating this
provider's ability to balance requests across multiple providers.

Parameters:

	feature: The name of the feature to check

Returns:

	bool: True if the feature is supported, false otherwise
*/
func (bp *BalancedProvider) SupportsFeature(feature string) bool {
	return feature == "balanced"
}

/*
ValidateConfig checks the validity of the configuration across all providers.
It ensures that each provider's configuration is valid for its specific
implementation.

Returns:

	error: Any error that occurred during validation
*/
func (bp *BalancedProvider) ValidateConfig() error {
	for _, p := range bp.providers {
		if err := p.provider.ValidateConfig(); err != nil {
			return err
		}
	}
	return nil
}

/*
Version returns the semantic version of the BalancedProvider implementation.

Returns:

	string: The version string in semantic versioning format
*/
func (bp *BalancedProvider) Version() string {
	return "1.0.0"
}

/*
AddProvider adds a new provider to the pool.
*/
func (lb *BalancedProvider) AddProvider(provider Provider) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	lb.providers = append(lb.providers, &ProviderStatus{
		provider:      provider,
		cooldownUntil: time.Time{},
		errorCount:    0,
	})
}

/*
InitializeProviders sets up the initial provider pool based on environment variables.
It attempts to initialize providers for different LLM services (OpenAI, Anthropic, etc.)
using their respective API keys from environment variables.

Returns:

	error: Any error that occurred during provider initialization
*/
func (bp *BalancedProvider) InitializeProviders() error {
	v := viper.GetViper()

	if v.GetBool("providers.openai") {
		if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey != "" {
			bp.AddProvider(NewOpenAI(openaiKey))
		}
	}

	if v.GetBool("providers.anthropic") {
		if anthropicKey := os.Getenv("ANTHROPIC_API_KEY"); anthropicKey != "" {
			bp.AddProvider(NewAnthropic(anthropicKey))
		}
	}

	if v.GetBool("providers.gemini") {
		if geminiKey := os.Getenv("GEMINI_API_KEY"); geminiKey != "" {
			bp.AddProvider(NewGemini(geminiKey))
		}
	}

	if v.GetBool("providers.cohere") {
		if cohereKey := os.Getenv("COHERE_API_KEY"); cohereKey != "" {
			bp.AddProvider(NewCohere(cohereKey))
		}
	}

	if v.GetBool("providers.deepseek") {
		if deepseekKey := os.Getenv("DEEPSEEK_API_KEY"); deepseekKey != "" {
			bp.AddProvider(NewDeepSeek(deepseekKey))
		}
	}

	if v.GetBool("providers.ollama") {
		bp.AddProvider(NewOllama("http://localhost:11434"))
	}

	if v.GetBool("providers.nvidia") {
		if nvidiaKey := os.Getenv("NVIDIA_API_KEY"); nvidiaKey != "" {
			bp.AddProvider(NewOpenAICompatible(
				nvidiaKey,
				v.GetString("endpoints.nvidia"),
				v.GetString("models.nvidia"),
				"nvidia",
			))
		}
	}

	if v.GetBool("providers.lmstudio") {
		if lmstudioKey := os.Getenv("LMSTUDIO_API_KEY"); lmstudioKey != "" {
			bp.AddProvider(NewOpenAICompatible(
				lmstudioKey,
				v.GetString("endpoints.lmstudio"),
				v.GetString("models.lmstudio"),
				"lmstudio",
			))
		}
	}

	// Check if any providers were added
	if len(bp.providers) == 0 {
		return errors.New("no providers were initialized")
	}

	return nil
}

/*
Read implements io.Reader. It reads data from the internal buffer into p.
This method is thread-safe and will return an error if the buffer is closed.

Parameters:

	p: The byte slice to read data into

Returns:

	int: The number of bytes read
	error: Any error that occurred during reading
*/
func (bp *BalancedProvider) Read(p []byte) (int, error) {
	bp.mu.RLock()
	defer bp.mu.RUnlock()
	if bp.rwBuffer == nil {
		return 0, errors.New("read/write buffer is closed")
	}
	return bp.rwBuffer.Read(p)
}

/*
Write implements io.Writer. It writes data from p into the internal buffer.
This method is thread-safe and will return an error if the buffer is closed.

Parameters:

	p: The byte slice containing data to write

Returns:

	int: The number of bytes written
	error: Any error that occurred during writing
*/
func (bp *BalancedProvider) Write(p []byte) (int, error) {
	bp.mu.RLock()
	defer bp.mu.RUnlock()
	if bp.rwBuffer == nil {
		return 0, errors.New("read/write buffer is closed")
	}
	return bp.rwBuffer.Write(p)
}

/*
Close implements io.Closer. It closes the internal buffer and cancels any ongoing
generation. This method is thread-safe and will return an error if the buffer
is already closed.

Returns:

	error: Any error that occurred during closing
*/
func (bp *BalancedProvider) Close() error {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	if bp.rwBuffer == nil {
		return errors.New("read/write buffer already closed")
	}
	bp.rwBuffer = nil
	return nil
}
