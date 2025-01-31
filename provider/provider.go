package provider

import (
	"context"
	"errors"
	"time"

	"github.com/theapemachine/caramba/utils"
)

// BasicGenerationParams contains basic parameters for generation requests
type BasicGenerationParams struct {
	Prompt      string
	MaxTokens   int
	Temperature float64
	TopP        float64
	StopTokens  []string
	Metadata    map[string]interface{}
}

// RateLimitConfig defines rate limiting parameters
type RateLimitConfig struct {
	RequestsPerSecond float64
	BurstSize         int
	Enabled           bool
}

// TimeoutConfig defines timeout parameters
type TimeoutConfig struct {
	RequestTimeout time.Duration
	ConnectTimeout time.Duration
	IdleTimeout    time.Duration
}

// RetryConfig defines retry behavior
type RetryConfig struct {
	MaxAttempts     int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	BackoffFactor   float64
	RetryableErrors []string
}

// AuthConfig contains authentication configuration
type AuthConfig struct {
	APIKey      string
	Credentials map[string]string
	AuthType    string
}

// FeatureFlags contains provider-specific feature toggles
type FeatureFlags struct {
	EnableStreaming bool
	EnableCache     bool
	DebugMode       bool
	CustomFeatures  map[string]interface{}
}

// ProviderConfig contains all provider configuration
type ProviderConfig struct {
	RateLimit    RateLimitConfig
	Timeout      TimeoutConfig
	RetryPolicy  RetryConfig
	Auth         AuthConfig
	Features     FeatureFlags
	CustomConfig map[string]interface{}
}

// ProviderMetrics contains provider performance metrics
type ProviderMetrics struct {
	RequestCount     int64
	SuccessCount     int64
	ErrorCount       int64
	AverageLatency   time.Duration
	TokensGenerated  int64
	CurrentRateLimit float64
	CustomMetrics    map[string]interface{}
}

// Provider defines the enhanced interface for AI providers
type Provider interface {
	// Core functionality
	Generate(context.Context, *LLMGenerationParams) <-chan *Event
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

// BaseProvider provides a default implementation of common provider functionality
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

// Configure implements basic configuration management
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

// GetConfig returns the current configuration
func (bp *BaseProvider) GetConfig() *ProviderConfig {
	return &bp.config
}

// GetMetrics returns current metrics
func (bp *BaseProvider) GetMetrics() (*ProviderMetrics, error) {
	return &bp.metrics, nil
}

// HealthCheck performs a basic health check
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
