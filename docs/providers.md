# Provider Management

The Provider Management system in Caramba handles interactions with multiple AI providers, implementing sophisticated load balancing, failover, and recovery mechanisms.

## Supported Providers

- OpenAI (GPT-4o-mini)
- Anthropic (Claude-3-5-sonnet)
- Google (Gemini-1.5-flash)
- Cohere (Command-r)
- HuggingFace (GPT2)
- NVIDIA (Llama-3.1-nemotron-70b)
- Ollama (llama3.2:3b)
- LMStudio (Llama-3.1-8B-Lexi)

## Architecture

### Provider Interface

```go
type Provider interface {
    Generate(context.Context, *GenerationParams) <-chan Event
    Name() string
}
```

### Balanced Provider

The `BalancedProvider` implements smart load balancing across multiple providers:

- Health monitoring
- Automatic failover
- Cooldown periods
- Thread-safe operations

## Configuration

### Environment Variables

```bash
# Required Provider Keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export COHERE_API_KEY="your-key"

# Optional Provider Keys
export HF_API_KEY="your-key"
export NVIDIA_API_KEY="your-key"
```

### Provider Status

```go
type ProviderStatus struct {
    provider Provider
    occupied bool
    lastUsed time.Time
    failures int
}
```

## Features

### Load Balancing

- Round-robin distribution
- Failure-aware routing
- Occupancy tracking
- Dynamic provider selection

### Health Monitoring

- Failure tracking
- Automatic recovery
- Cooldown periods
- Health status reporting

### Thread Safety

- Mutex-based synchronization
- Atomic operations
- Race condition prevention
- Safe provider selection

### Error Handling

- Automatic retries
- Graceful degradation
- Error event propagation
- Provider-specific error handling

## Usage

### Basic Usage

```go
provider := provider.NewBalancedProvider()
params := provider.NewGenerationParams()
response := provider.Generate(context.Background(), params)

for event := range response {
    // Handle response events
}
```

### Custom Provider Configuration

```go
provider := provider.NewOpenAICompatible(
    os.Getenv("CUSTOM_API_KEY"),
    "https://custom-endpoint",
    "custom-model",
)
```

## Best Practices

1. **Provider Configuration**

   - Configure multiple providers when possible
   - Set appropriate API keys
   - Monitor rate limits
   - Configure model parameters

2. **Error Handling**

   - Implement proper retry logic
   - Handle provider-specific errors
   - Monitor provider health
   - Implement graceful degradation

3. **Performance Optimization**

   - Monitor provider latency
   - Implement request caching
   - Optimize request parameters
   - Balance load appropriately

4. **Security**
   - Secure API key storage
   - Implement request logging
   - Monitor usage patterns
   - Implement access controls

## Advanced Features

### Custom Providers

```go
type CustomProvider struct {
    // Provider implementation
}

func (cp *CustomProvider) Generate(ctx context.Context, params *GenerationParams) <-chan Event {
    // Implementation
}

func (cp *CustomProvider) Name() string {
    return "custom"
}
```

### Provider Selection Strategy

```go
type SelectionStrategy interface {
    SelectProvider([]*ProviderStatus) *ProviderStatus
}
```

## Troubleshooting

Common issues and solutions:

1. **Rate Limiting**

   - Implement exponential backoff
   - Monitor request frequency
   - Use multiple providers
   - Cache responses when possible

2. **Provider Failures**

   - Check API credentials
   - Verify network connectivity
   - Monitor provider status
   - Check request parameters

3. **Performance Issues**
   - Monitor response times
   - Check provider load
   - Optimize request size
   - Implement caching

## Future Development

Planned improvements:

- Enhanced provider selection algorithms
- Advanced caching mechanisms
- Improved error recovery
- Extended provider support
- Performance optimizations
- Advanced monitoring capabilities
