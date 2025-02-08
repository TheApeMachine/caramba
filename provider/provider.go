package provider

import (
	"context"
	"errors"
	"time"

	"github.com/theapemachine/caramba/utils"
)

/*
BasicGenerationParams contains basic parameters for
generation requests.
*/
type BasicGenerationParams struct {
	Prompt      string
	MaxTokens   int
	Temperature float64
	TopP        float64
	StopTokens  []string
	Metadata    map[string]interface{}
}

/*
RateLimitConfig defines rate limiting parameters for provider requests.
It controls the rate at which requests can be made to prevent overloading
and ensure fair resource usage.
*/
type RateLimitConfig struct {
	RequestsPerSecond float64
	BurstSize         int
	Enabled           bool
}

/*
TimeoutConfig defines timeout parameters for various operations.
It specifies duration limits for different types of timeouts to ensure
proper request handling and resource cleanup.
*/
type TimeoutConfig struct {
	RequestTimeout time.Duration
	ConnectTimeout time.Duration
	IdleTimeout    time.Duration
}

/*
RetryConfig defines retry behavior for failed operations.
It specifies how many times to retry, how long to wait between retries,
and what types of errors are retryable.
*/
type RetryConfig struct {
	MaxAttempts     int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	BackoffFactor   float64
	RetryableErrors []string
}

/*
AuthConfig contains authentication configuration for provider access.
It stores various authentication methods and credentials needed to
interact with the provider's API.
*/
type AuthConfig struct {
	APIKey      string
	Credentials map[string]string
	AuthType    string
}

/*
FeatureFlags contains provider-specific feature toggles.
It controls various optional features and behaviors of the provider,
allowing for flexible configuration of provider capabilities.
*/
type FeatureFlags struct {
	EnableStreaming bool
	EnableCache     bool
	DebugMode       bool
	CustomFeatures  map[string]interface{}
}

/*
ProviderConfig contains all provider configuration options.
It aggregates various configuration components including rate limiting,
timeouts, retry policies, authentication, and feature flags.
*/
type ProviderConfig struct {
	RateLimit    RateLimitConfig
	Timeout      TimeoutConfig
	RetryPolicy  RetryConfig
	Auth         AuthConfig
	Features     FeatureFlags
	CustomConfig map[string]interface{}
}

/*
ProviderMetrics contains provider performance metrics and statistics.
It tracks various operational metrics including request counts,
success rates, latency, and provider-specific custom metrics.
*/
type ProviderMetrics struct {
	RequestCount     int64
	SuccessCount     int64
	ErrorCount       int64
	AverageLatency   time.Duration
	TokensGenerated  int64
	CurrentRateLimit float64
	CustomMetrics    map[string]interface{}
}

/*
Provider defines the enhanced interface for AI providers.
It specifies the complete set of operations that must be implemented
by any provider, including core functionality, configuration management,
monitoring, resource management, and capability introspection.

Interface Methods:

	Core Functionality:
	    - Generate: Produces AI-generated content
	    - Name: Returns provider identifier
	    - Version: Returns provider version

	Configuration:
	    - Configure: Applies provider configuration
	    - GetConfig: Retrieves current configuration
	    - ValidateConfig: Validates configuration

	Monitoring:
	    - GetMetrics: Retrieves performance metrics
	    - HealthCheck: Checks provider health status

	Resource Management:
	    - Initialize: Sets up provider resources
	    - Cleanup: Cleans up provider resources

	Stream Management:
	    - PauseGeneration: Pauses content generation
	    - ResumeGeneration: Resumes content generation
	    - CancelGeneration: Cancels ongoing generation

	Capabilities:
	    - GetCapabilities: Lists provider capabilities
	    - SupportsFeature: Checks feature support
*/
type Provider interface {
	// Core functionality
	Generate(*LLMGenerationParams) <-chan *Event
	Name() string
	Version() string

	// Configuration management
	Configure(*ProviderConfig) error
	GetConfig() *ProviderConfig
	ValidateConfig() error

	// Monitoring and metrics
	GetMetrics() (*ProviderMetrics, error)
	HealthCheck(context.Context) *utils.HealthStatus

	// Resource management
	Initialize(context.Context) error
	Cleanup(context.Context) error

	// Stream management
	PauseGeneration() error
	ResumeGeneration() error
	CancelGeneration(context.Context) error

	// Capability introspection
	GetCapabilities() map[string]interface{}
	SupportsFeature(feature string) bool
}

/*
BaseProvider provides a default implementation of common provider functionality.
It implements basic versions of the Provider interface methods and can be
embedded in specific provider implementations to reduce boilerplate code.
*/
type BaseProvider struct {
	config   ProviderConfig
	metrics  ProviderMetrics
	features map[string]interface{}
}

// NewBaseProvider creates a new BaseProvider with default configuration
func NewBaseProvider() *BaseProvider {
	return &BaseProvider{
		config: ProviderConfig{
			RateLimit: RateLimitConfig{
				RequestsPerSecond: 10,
				BurstSize:         20,
				Enabled:           true,
			},
			Timeout: TimeoutConfig{
				RequestTimeout: 30 * time.Second,
				ConnectTimeout: 10 * time.Second,
				IdleTimeout:    60 * time.Second,
			},
			RetryPolicy: RetryConfig{
				MaxAttempts:   3,
				InitialDelay:  100 * time.Millisecond,
				MaxDelay:      5 * time.Second,
				BackoffFactor: 2.0,
			},
			Features: FeatureFlags{
				EnableStreaming: true,
				EnableCache:     false,
				DebugMode:       false,
			},
		},
		metrics:  ProviderMetrics{},
		features: make(map[string]interface{}),
	}
}

/*
Configure implements basic configuration management for the provider.
It validates and applies the provided configuration to the base provider.

Parameters:

	config: The provider configuration to apply

Returns:

	error: An error if the configuration is invalid or nil if successful
*/
func (bp *BaseProvider) Configure(config *ProviderConfig) error {
	if config == nil {
		return utils.NewErrorWithContext(
			ErrInvalidConfig,
			utils.SeverityHigh,
		).WithContext("reason", "nil config")
	}
	bp.config = *config
	return nil
}

/*
GetConfig returns the current provider configuration.

Returns:

	*ProviderConfig: A pointer to the current configuration
*/
func (bp *BaseProvider) GetConfig() *ProviderConfig {
	return &bp.config
}

/*
GetMetrics returns current provider metrics.

Returns:

	*ProviderMetrics: A pointer to the current metrics
	error: Any error that occurred while gathering metrics
*/
func (bp *BaseProvider) GetMetrics() (*ProviderMetrics, error) {
	return &bp.metrics, nil
}

/*
HealthCheck performs a basic health check on the provider.
It evaluates the provider's health based on the error rate.

Parameters:

	ctx: Context for the health check operation

Returns:

	*utils.HealthStatus: A pointer to either StatusHealthy or StatusDegraded
*/
func (bp *BaseProvider) HealthCheck(ctx context.Context) *utils.HealthStatus {
	status := utils.StatusHealthy
	if bp.metrics.ErrorCount > bp.metrics.SuccessCount {
		status = utils.StatusDegraded
	}
	return &status
}

// Common provider errors
var (
	ErrInvalidConfig  = errors.New("invalid provider configuration")
	ErrNotInitialized = errors.New("provider not initialized")
	ErrNotImplemented = errors.New("method not implemented")
	ErrInvalidParams  = errors.New("invalid generation parameters")
	ErrRateLimited    = errors.New("rate limit exceeded")
)
