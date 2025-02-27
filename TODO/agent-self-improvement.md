# Agent Self-Improvement Capabilities

## 2.1 Performance Metrics System

**New Files to Create:**
- `pkg/agent/metrics/metrics.go`
- `pkg/agent/metrics/collector.go`

**Files to Modify:**
- `pkg/agent/core/base_agent.go`
- `pkg/agent/core/interfaces.go`

**Implementation Details:**

1. Create the metrics interface in `metrics.go`:

```go
package metrics

import (
    "context"
    "time"
)

// MetricsCollector defines the interface for collecting agent metrics
type MetricsCollector interface {
    // RecordExecutionTime records the time taken for an execution
    RecordExecutionTime(agentID string, executionTime time.Duration)
    
    // RecordTokenUsage records token usage for an execution
    RecordTokenUsage(agentID string, promptTokens, completionTokens int)
    
    // RecordToolUsage records tool usage statistics
    RecordToolUsage(agentID, toolName string, successful bool, executionTime time.Duration)
    
    // RecordMemoryUsage records memory usage statistics
    RecordMemoryUsage(agentID string, retrieved, stored int)
    
    // RecordCompletionQuality records the quality of an agent's completion
    RecordCompletionQuality(agentID string, quality float32, source string)
    
    // GetAgentMetrics retrieves metrics for a specific agent
    GetAgentMetrics(agentID string) (*AgentMetrics, error)
    
    // GetSystemMetrics retrieves system-wide metrics
    GetSystemMetrics() (*SystemMetrics, error)
}

// AgentMetrics contains performance metrics for a single agent
type AgentMetrics struct {
    AgentID                 string
    TotalExecutions         int
    AverageExecutionTime    time.Duration
    TotalTokensUsed         int
    ToolUsage               map[string]ToolMetrics
    MemoryRetrievals        int
    MemoryStores            int
    SuccessRate             float32
    AverageCompletionQuality float32
    LLMParameters           map[string]interface{}
}

// ToolMetrics contains metrics for tool usage
type ToolMetrics struct {
    Uses                int
    SuccessfulUses      int
    AverageExecutionTime time.Duration
}

// SystemMetrics contains system-wide performance metrics
type SystemMetrics struct {
    TotalExecutions     int
    TotalTokensUsed     int
    AverageExecutionTime time.Duration
    AgentCount          int
    StartTime           time.Time
    TopPerformingAgents []string
    MostUsedTools       map[string]int
}
```

2. Implement the metrics collector in `collector.go`:

```go
package metrics

import (
    "sync"
    "time"
)

// InMemoryMetricsCollector implements the MetricsCollector interface in memory
type InMemoryMetricsCollector struct {
    mu sync.RWMutex
    
    // Agent metrics
    agentMetrics map[string]*AgentMetricsData
    
    // System metrics
    startTime       time.Time
    totalExecutions int
    totalTokens     int
    totalTime       time.Duration
    
    // Tool usage
    toolUsage map[string]int
}

// AgentMetricsData contains detailed metrics data for an agent
type AgentMetricsData struct {
    AgentID             string
    Executions          int
    TotalExecutionTime  time.Duration
    PromptTokens        int
    CompletionTokens    int
    ToolUsage           map[string]*ToolMetricsData
    MemoryRetrievals    int
    MemoryStores        int
    QualityScores       []float32
    QualitySources      map[string][]float32
    Parameters          map[string]interface{}
}

// ToolMetricsData contains detailed metrics for tool usage
type ToolMetricsData struct {
    Uses                int
    SuccessfulUses      int
    TotalExecutionTime  time.Duration
}

// NewInMemoryMetricsCollector creates a new in-memory metrics collector
func NewInMemoryMetricsCollector() *InMemoryMetricsCollector {
    return &InMemoryMetricsCollector{
        agentMetrics: make(map[string]*AgentMetricsData),
        startTime:    time.Now(),
        toolUsage:    make(map[string]int),
    }
}

// RecordExecutionTime records the time taken for an execution
func (c *InMemoryMetricsCollector) RecordExecutionTime(agentID string, executionTime time.Duration) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    // Update agent metrics
    agent := c.getOrCreateAgentMetrics(agentID)
    agent.Executions++
    agent.TotalExecutionTime += executionTime
    
    // Update system metrics
    c.totalExecutions++
    c.totalTime += executionTime
}

// RecordTokenUsage records token usage for an execution
func (c *InMemoryMetricsCollector) RecordTokenUsage(agentID string, promptTokens, completionTokens int) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    agent := c.getOrCreateAgentMetrics(agentID)
    agent.PromptTokens += promptTokens
    agent.CompletionTokens += completionTokens
    
    c.totalTokens += (promptTokens + completionTokens)
}

// RecordToolUsage records tool usage statistics
func (c *InMemoryMetricsCollector) RecordToolUsage(agentID, toolName string, successful bool, executionTime time.Duration) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    // Update agent's tool metrics
    agent := c.getOrCreateAgentMetrics(agentID)
    if agent.ToolUsage == nil {
        agent.ToolUsage = make(map[string]*ToolMetricsData)
    }
    
    toolMetrics, exists := agent.ToolUsage[toolName]
    if !exists {
        toolMetrics = &ToolMetricsData{}
        agent.ToolUsage[toolName] = toolMetrics
    }
    
    toolMetrics.Uses++
    if successful {
        toolMetrics.SuccessfulUses++
    }
    toolMetrics.TotalExecutionTime += executionTime
    
    // Update global tool usage
    c.toolUsage[toolName]++
}

// RecordMemoryUsage records memory usage statistics
func (c *InMemoryMetricsCollector) RecordMemoryUsage(agentID string, retrieved, stored int) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    agent := c.getOrCreateAgentMetrics(agentID)
    agent.MemoryRetrievals += retrieved
    agent.MemoryStores += stored
}

// RecordCompletionQuality records the quality of an agent's completion
func (c *InMemoryMetricsCollector) RecordCompletionQuality(agentID string, quality float32, source string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    agent := c.getOrCreateAgentMetrics(agentID)
    agent.QualityScores = append(agent.QualityScores, quality)
    
    if agent.QualitySources == nil {
        agent.QualitySources = make(map[string][]float32)
    }
    agent.QualitySources[source] = append(agent.QualitySources[source], quality)
}

// GetAgentMetrics retrieves metrics for a specific agent
func (c *InMemoryMetricsCollector) GetAgentMetrics(agentID string) (*AgentMetrics, error) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    data, exists := c.agentMetrics[agentID]
    if !exists {
        return nil, fmt.Errorf("no metrics found for agent %s", agentID)
    }
    
    metrics := &AgentMetrics{
        AgentID:             data.AgentID,
        TotalExecutions:     data.Executions,
        TotalTokensUsed:     data.PromptTokens + data.CompletionTokens,
        MemoryRetrievals:    data.MemoryRetrievals,
        MemoryStores:        data.MemoryStores,
        LLMParameters:       data.Parameters,
    }
    
    // Calculate averages
    if data.Executions > 0 {
        metrics.AverageExecutionTime = data.TotalExecutionTime / time.Duration(data.Executions)
    }
    
    // Calculate quality
    if len(data.QualityScores) > 0 {
        var sum float32
        for _, score := range data.QualityScores {
            sum += score
        }
        metrics.AverageCompletionQuality = sum / float32(len(data.QualityScores))
    }
    
    // Calculate tool metrics
    metrics.ToolUsage = make(map[string]ToolMetrics)
    for name, tool := range data.ToolUsage {
        avgTime := time.Duration(0)
        if tool.Uses > 0 {
            avgTime = tool.TotalExecutionTime / time.Duration(tool.Uses)
        }
        
        metrics.ToolUsage[name] = ToolMetrics{
            Uses:                tool.Uses,
            SuccessfulUses:      tool.SuccessfulUses,
            AverageExecutionTime: avgTime,
        }
    }
    
    // Calculate success rate
    metrics.SuccessRate = 1.0 // Default perfect
    
    totalToolUses := 0
    successfulToolUses := 0
    for _, tool := range data.ToolUsage {
        totalToolUses += tool.Uses
        successfulToolUses += tool.SuccessfulUses
    }
    
    if totalToolUses > 0 {
        metrics.SuccessRate = float32(successfulToolUses) / float32(totalToolUses)
    }
    
    return metrics, nil
}

// GetSystemMetrics retrieves system-wide metrics
func (c *InMemoryMetricsCollector) GetSystemMetrics() (*SystemMetrics, error) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    avgTime := time.Duration(0)
    if c.totalExecutions > 0 {
        avgTime = c.totalTime / time.Duration(c.totalExecutions)
    }
    
    // Find top performing agents
    type agentPerf struct {
        id    string
        score float32
    }
    
    var performances []agentPerf
    for id, agent := range c.agentMetrics {
        var score float32 = 0
        
        // Calculate a composite score based on quality and success rate
        if len(agent.QualityScores) > 0 {
            var sum float32
            for _, s := range agent.QualityScores {
                sum += s
            }
            score += sum / float32(len(agent.QualityScores))
        }
        
        totalToolUses := 0
        successfulToolUses := 0
        for _, tool := range agent.ToolUsage {
            totalToolUses += tool.Uses
            successfulToolUses += tool.SuccessfulUses
        }
        
        if totalToolUses > 0 {
            successRate := float32(successfulToolUses) / float32(totalToolUses)
            score += successRate
        }
        
        // Scale by number of executions (more executions = more reliable data)
        if agent.Executions > 0 {
            score = score * float32(math.Log1p(float64(agent.Executions)))
        }
        
        performances = append(performances, agentPerf{id, score})
    }
    
    // Sort by score
    sort.Slice(performances, func(i, j int) bool {
        return performances[i].score > performances[j].score
    })
    
    // Get top 5 or fewer
    topCount := 5
    if len(performances) < topCount {
        topCount = len(performances)
    }
    
    topAgents := make([]string, topCount)
    for i := 0; i < topCount; i++ {
        topAgents[i] = performances[i].id
    }
    
    // Get most used tools
    type toolUsage struct {
        name  string
        count int
    }
    
    var toolUsages []toolUsage
    for name, count := range c.toolUsage {
        toolUsages = append(toolUsages, toolUsage{name, count})
    }
    
    sort.Slice(toolUsages, func(i, j int) bool {
        return toolUsages[i].count > toolUsages[j].count
    })
    
    // Get top 10 or fewer
    toolCount := 10
    if len(toolUsages) < toolCount {
        toolCount = len(toolUsages)
    }
    
    mostUsedTools := make(map[string]int, toolCount)
    for i := 0; i < toolCount; i++ {
        mostUsedTools[toolUsages[i].name] = toolUsages[i].count
    }
    
    return &SystemMetrics{
        TotalExecutions:     c.totalExecutions,
        TotalTokensUsed:     c.totalTokens,
        AverageExecutionTime: avgTime,
        AgentCount:          len(c.agentMetrics),
        StartTime:           c.startTime,
        TopPerformingAgents: topAgents,
        MostUsedTools:       mostUsedTools,
    }, nil
}

// getOrCreateAgentMetrics gets or creates agent metrics data
func (c *InMemoryMetricsCollector) getOrCreateAgentMetrics(agentID string) *AgentMetricsData {
    data, exists := c.agentMetrics[agentID]
    if !exists {
        data = &AgentMetricsData{
            AgentID:    agentID,
            Parameters: make(map[string]interface{}),
            ToolUsage:  make(map[string]*ToolMetricsData),
        }
        c.agentMetrics[agentID] = data
    }
    return data
}
```

3. Modify `base_agent.go` to use metrics:

```go
// Execute runs the agent with the provided input and returns a response
func (a *BaseAgent) Execute(ctx context.Context, message core.LLMMessage) (string, error) {
    startTime := time.Now()
    
    if a.LLM == nil {
        return "", errors.New("no LLM provider set")
    }

    // Log the execution start with summarized input
    output.Action("agent", "execute", fmt.Sprintf("%s processing: %s", a.Name, output.Summarize(message.Content, 60)))

    // Check for memory integration and enhance context if available
    memorySpinner := output.StartSpinner("Checking memory for relevant context")
    enhancedMessage := a.injectMemories(ctx, message)
    message = enhancedMessage // Use the enhanced message
    output.StopSpinner(memorySpinner, "Memory retrieval complete")
    
    // Record memory retrieval in metrics if available
    if a.Metrics != nil {
        a.Metrics.RecordMemoryUsage(a.Name, 1, 0) // Simplified: count as 1 retrieval
    }

    // Create a plan if a planner is available
    if a.Planner != nil {
        planSpinner := output.StartSpinner("Planning execution strategy")
        _, err := a.createPlan(ctx, message)
        if err != nil {
            output.StopSpinner(planSpinner, "")
            output.Error("Planning failed", err)
        } else {
            output.StopSpinner(planSpinner, "Execution plan created")
        }
    }

    // Prepare for iterations
    var response strings.Builder
    iteration := 0
    var totalPromptTokens, totalCompletionTokens int

    // Format the message for the first iteration
    iterMsg := core.LLMMessage{
        Role:    message.Role,
        Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, message.Content),
    }

    // Run the iterations
    for iteration < a.IterationLimit {
        // Add the current message to the conversation
        a.Params.Messages = append(a.Params.Messages, iterMsg)

        // IMPORTANT: Copy tools to LLM parameters before making the API call
        a.toolsMu.RLock()
        a.Params.Tools = a.Tools
        a.toolsMu.RUnlock()

        // Show thinking spinner
        thinkingSpinner := output.StartSpinner(fmt.Sprintf("Iteration %d/%d: Agent thinking", iteration+1, a.IterationLimit))

        // Generate the response
        iterStartTime := time.Now()
        res := a.LLM.GenerateResponse(ctx, a.Params)
        iterDuration := time.Since(iterStartTime)

        // Record metrics for token usage if available
        if a.Metrics != nil && res.TokenUsage != nil {
            a.Metrics.RecordTokenUsage(a.Name, 
                res.TokenUsage.PromptTokens, 
                res.TokenUsage.CompletionTokens)
            
            totalPromptTokens += res.TokenUsage.PromptTokens
            totalCompletionTokens += res.TokenUsage.CompletionTokens
        }

        // Handle errors
        if res.Error != nil {
            output.StopSpinner(thinkingSpinner, "")
            output.Error("LLM response generation failed", res.Error)
            return "", res.Error
        }

        // Success - stop spinner with appropriate message
        if len(res.ToolCalls) > 0 {
            output.StopSpinner(thinkingSpinner, fmt.Sprintf("Agent is using %d tools", len(res.ToolCalls)))

            // Log each tool call and execute tools
            for _, toolCall := range res.ToolCalls {
                output.Action("agent", "tool_call", toolCall.Name)
                
                // Execute the tool and measure performance
                toolStartTime := time.Now()
                toolResult, toolErr := a.executeTool(ctx, toolCall)
                toolDuration := time.Since(toolStartTime)
                
                // Record tool usage metrics
                if a.Metrics != nil {
                    a.Metrics.RecordToolUsage(a.Name, toolCall.Name, toolErr == nil, toolDuration)
                }
                
                // Handle tool result...
            }
        } else {
            output.StopSpinner(thinkingSpinner, "Agent completed thinking")
        }

        // Add the response to the accumulated response
        response.WriteString(res.Content)

        // Next iteration
        iteration++

        // Prepare the next message as assistant's response
        iterMsg = core.LLMMessage{
            Role:    "assistant",
            Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, response.String()),
        }
    }

    // Build context window for memory extraction
    var contextWindow strings.Builder
    for _, message := range a.Params.Messages {
        switch message.Role {
        case "user":
            contextWindow.WriteString(fmt.Sprintf("User: %s\n", message.Content))
        case "assistant":
            contextWindow.WriteString(fmt.Sprintf("Assistant: %s\n", message.Content))
        }
    }

    // Extract memories from the conversation
    memExtractSpinner := output.StartSpinner("Processing and storing memories")
    memories, err := a.extractMemories(ctx, contextWindow.String())
    if err != nil {
        output.Warn(fmt.Sprintf("Memory extraction warning: %v", err))
    }
    output.StopSpinner(memExtractSpinner, "New memories stored")
    
    // Record memory storage in metrics
    if a.Metrics != nil && len(memories) > 0 {
        a.Metrics.RecordMemoryUsage(a.Name, 0, len(memories))
    }

    // Record total execution time
    totalTime := time.Since(startTime)
    if a.Metrics != nil {
        a.Metrics.RecordExecutionTime(a.Name, totalTime)
    }

    // Final reporting
    output.Result(fmt.Sprintf("Agent %s completed execution (%d tokens, %v)", 
        a.Name, totalPromptTokens + totalCompletionTokens, totalTime.Round(time.Millisecond)))

    return response.String(), nil
}

// executeTool is a helper function to execute a tool
func (a *BaseAgent) executeTool(ctx context.Context, toolCall core.ToolCall) (interface{}, error) {
    // Find the tool
    var tool core.Tool
    for _, t := range a.Tools {
        if t.Name() == toolCall.Name {
            tool = t
            break
        }
    }
    
    if tool == nil {
        return nil, fmt.Errorf("tool not found: %s", toolCall.Name)
    }
    
    // Execute the tool
    return tool.Execute(ctx, toolCall.Args)
}
```

## 2.2 Parameter Optimization System

**New Files to Create:**
- `pkg/agent/optimization/optimizer.go`
- `pkg/agent/optimization/strategies.go`

**Implementation Details:**

1. Create the optimizer interface in `optimizer.go`:

```go
package optimization

import (
    "context"
    "time"
    
    "github.com/theapemachine/caramba/pkg/agent/core"
    "github.com/theapemachine/caramba/pkg/agent/metrics"
)

// Optimizer is responsible for tuning agent parameters to improve performance
type Optimizer interface {
    // OptimizeAgent optimizes an agent's parameters based on performance metrics
    OptimizeAgent(ctx context.Context, agent core.Agent) error
    
    // GetOptimizationHistory returns the history of optimization attempts
    GetOptimizationHistory(agentID string) ([]OptimizationEntry, error)
    
    // SetStrategy sets the optimization strategy
    SetStrategy(strategy OptimizationStrategy)
}

// OptimizationStrategy defines the algorithm used for parameter optimization
type OptimizationStrategy interface {
    // Name returns the name of the strategy
    Name() string
    
    // Optimize performs a single optimization step
    Optimize(ctx context.Context, agent core.Agent, metrics *metrics.AgentMetrics) (map[string]interface{}, error)
}

// OptimizationEntry represents a single optimization attempt
type OptimizationEntry struct {
    Timestamp  time.Time
    AgentID    string
    Strategy   string
    OldParams  map[string]interface{}
    NewParams  map[string]interface{}
    Metrics    *metrics.AgentMetrics
    Results    *OptimizationResults
}

// OptimizationResults contains the results of an optimization attempt
type OptimizationResults struct {
    Success          bool
    MetricsBefore    map[string]float64
    MetricsAfter     map[string]float64
    PercentImprovement map[string]float64
}

// BasicOptimizer implements the Optimizer interface
type BasicOptimizer struct {
    metrics      metrics.MetricsCollector
    strategy     OptimizationStrategy
    history      map[string][]OptimizationEntry
    minInterval  time.Duration
    lastRun      map[string]time.Time
}

// NewBasicOptimizer creates a new basic optimizer
func NewBasicOptimizer(metricsCollector metrics.MetricsCollector) *BasicOptimizer {
    return &BasicOptimizer{
        metrics:     metricsCollector,
        strategy:    NewBayesianOptimizationStrategy(),
        history:     make(map[string][]OptimizationEntry),
        minInterval: 1 * time.Hour,
        lastRun:     make(map[string]time.Time),
    }
}

// OptimizeAgent optimizes an agent's parameters based on performance metrics
func (o *BasicOptimizer) OptimizeAgent(ctx context.Context, agent core.Agent) error {
    agentID := agent.GetID()
    
    // Check if we've optimized recently
    if lastTime, ok := o.lastRun[agentID]; ok {
        if time.Since(lastTime) < o.minInterval {
            return fmt.Errorf("optimizer cooldown period active (last run: %v)", lastTime)
        }
    }
    
    // Get current metrics
    metrics, err := o.metrics.GetAgentMetrics(agentID)
    if err != nil {
        return fmt.Errorf("failed to get agent metrics: %w", err)
    }
    
    // Check if we have enough data to optimize
    if metrics.TotalExecutions < 10 {
        return fmt.Errorf("not enough execution data for optimization (need 10, have %d)", metrics.TotalExecutions)
    }
    
    // Get current parameters
    oldParams := agent.GetParameters()
    
    // Run the optimization strategy
    newParams, err := o.strategy.Optimize(ctx, agent, metrics)
    if err != nil {
        return fmt.Errorf("optimization failed: %w", err)
    }
    
    // Apply the new parameters
    if err := agent.SetParameters(newParams); err != nil {
        return fmt.Errorf("failed to apply new parameters: %w", err)
    }
    
    // Record the optimization in history
    entry := OptimizationEntry{
        Timestamp: time.Now(),
        AgentID:   agentID,
        Strategy:  o.strategy.Name(),
        OldParams: oldParams,
        NewParams: newParams,
        Metrics:   metrics,
        Results:   nil, // Will be updated later after evaluation
    }
    
    o.history[agentID] = append(o.history[agentID], entry)
    o.lastRun[agentID] = time.Now()
    
    return nil
}

// GetOptimizationHistory returns the history of optimization attempts
func (o *BasicOptimizer) GetOptimizationHistory(agentID string) ([]OptimizationEntry, error) {
    history, ok := o.history[agentID]
    if !ok {
        return nil, fmt.Errorf("no optimization history for agent %s", agentID)
    }
    return history, nil
}

// SetStrategy sets the optimization strategy
func (o *BasicOptimizer) SetStrategy(strategy OptimizationStrategy) {
    o.strategy = strategy
}
```

2. Implement optimization strategies in `strategies.go`:

```go
package optimization

import (
    "context"
    "fmt"
    "math/rand"
    "time"
    
    "github.com/theapemachine/caramba/pkg/agent/core"
    "github.com/theapemachine/caramba/pkg/agent/metrics"
    "github.com/theapemachine/caramba/pkg/output"
)

// Parameter ranges and constraints
var (
    temperatureRange = [2]float64{0.0, 1.0}
    topPRange        = [2]float64{0.0, 1.0}
    maxTokensRange   = [2]int{256, 4096}
)

// RandomSearchStrategy implements a basic random search optimization
type RandomSearchStrategy struct {
    iterations int
}

// NewRandomSearchStrategy creates a new random search strategy
func NewRandomSearchStrategy(iterations int) *RandomSearchStrategy {
    if iterations <= 0 {
        iterations = 10
    }
    return &RandomSearchStrategy{
        iterations: iterations,
    }
}

// Name returns the name of the strategy
func (s *RandomSearchStrategy) Name() string {
    return "RandomSearch"
}

// Optimize performs random search optimization
func (s *RandomSearchStrategy) Optimize(ctx context.Context, agent core.Agent, metrics *metrics.AgentMetrics) (map[string]interface{}, error) {
    output.Verbose(fmt.Sprintf("Starting random search optimization for agent %s", agent.GetID()))
    
    // Get current parameters
    currentParams := agent.GetParameters()
    
    // Extract current values or use defaults
    currentTemp := getFloat64Param(currentParams, "temperature", 0.7)
    currentTopP := getFloat64Param(currentParams, "top_p", 1.0)
    currentMaxTokens := getIntParam(currentParams, "max_tokens", 1024)
    
    // Generate random candidates
    bestScore := evaluateParameters(metrics)
    bestParams := currentParams
    
    for i := 0; i < s.iterations; i++ {
        // Generate random variations
        candidateParams := map[string]interface{}{
            "temperature": randomInRange(temperatureRange[0], temperatureRange[1]),
            "top_p":       randomInRange(topPRange[0], topPRange[1]),
            "max_tokens":  randomIntInRange(maxTokensRange[0], maxTokensRange[1]),
        }
        
        // Respect original system prompt
        if systemPrompt, ok := currentParams["system_prompt"].(string); ok {
            candidateParams["system_prompt"] = systemPrompt
        }
        
        // Speculatively evaluate this parameter set
        score := simulateParameters(metrics, candidateParams)
        
        output.Debug(fmt.Sprintf("Random search iteration %d: score %.4f (temp=%.2f, top_p=%.2f, max_tokens=%d)",
            i+1, score, 
            candidateParams["temperature"], 
            candidateParams["top_p"], 
            candidateParams["max_tokens"]))
        
        if score > bestScore {
            bestScore = score
            bestParams = candidateParams
        }
    }
    
    // If best parameters are different from current, use them
    if bestParams["temperature"] != currentTemp || 
       bestParams["top_p"] != currentTopP || 
       bestParams["max_tokens"] != currentMaxTokens {
        output.Result(fmt.Sprintf("Found improved parameters for agent %s: temp=%.2f, top_p=%.2f, max_tokens=%d",
            agent.GetID(), 
            bestParams["temperature"], 
            bestParams["top_p"], 
            bestParams["max_tokens"]))
        return bestParams, nil
    }
    
    // No improvement
    output.Info(fmt.Sprintf("No parameter improvements found for agent %s", agent.GetID()))
    return currentParams, nil
}

// BayesianOptimizationStrategy implements Bayesian optimization
// For a real implementation, you would use a library like goptuna or goai
// This is a simplified placeholder
type BayesianOptimizationStrategy struct {
    iterations int
}

// NewBayesianOptimizationStrategy creates a new Bayesian optimization strategy
func NewBayesianOptimizationStrategy() *BayesianOptimizationStrategy {
    return &BayesianOptimizationStrategy{
        iterations: 20,
    }
}

// Name returns the name of the strategy
func (s *BayesianOptimizationStrategy) Name() string {
    return "BayesianOptimization"
}

// Optimize performs Bayesian optimization
func (s *BayesianOptimizationStrategy) Optimize(ctx context.Context, agent core.Agent, metrics *metrics.AgentMetrics) (map[string]interface{}, error) {
    // In a real implementation, this would use Bayesian optimization
    // For this placeholder, we'll just use a more focused random search
    
    output.Verbose(fmt.Sprintf("Starting Bayesian optimization for agent %s", agent.GetID()))
    
    // Get current parameters
    currentParams := agent.GetParameters()
    
    // Extract current values or use defaults
    currentTemp := getFloat64Param(currentParams, "temperature", 0.7)
    currentTopP := getFloat64Param(currentParams, "top_p", 1.0)
    currentMaxTokens := getIntParam(currentParams, "max_tokens", 1024)
    
    // Start with current parameters as best
    bestScore := evaluateParameters(metrics)
    bestParams := currentParams
    
    // Focus search around current values (exploration/exploitation tradeoff)
    for i := 0; i < s.iterations; i++ {
        // Generate variations focused around current best
        candidateParams := map[string]interface{}{
            "temperature": boundedValue(currentTemp + randomInRange(-0.2, 0.2), temperatureRange[0], temperatureRange[1]),
            "top_p":       boundedValue(currentTopP + randomInRange(-0.2, 0.2), topPRange[0], topPRange[1]),
            "max_tokens":  boundedIntValue(currentMaxTokens + randomIntInRange(-256, 256), maxTokensRange[0], maxTokensRange[1]),
        }
        
        // Respect original system prompt
        if systemPrompt, ok := currentParams["system_prompt"].(string); ok {
            candidateParams["system_prompt"] = systemPrompt
        }
        
        // Speculatively evaluate this parameter set
        score := simulateParameters(metrics, candidateParams)
        
        if score > bestScore {
            bestScore = score
            bestParams = candidateParams
            
            // Update current values to focus search
            currentTemp = bestParams["temperature"].(float64)
            currentTopP = bestParams["top_p"].(float64)
            currentMaxTokens = bestParams["max_tokens"].(int)
        }
    }
    
    // If best parameters are different from original, use them
    originalTemp := getFloat64Param(agent.GetParameters(), "temperature", 0.7)
    originalTopP := getFloat64Param(agent.GetParameters(), "top_p", 1.0)
    originalMaxTokens := getIntParam(agent.GetParameters(), "max_tokens", 1024)
    
    if bestParams["temperature"] != originalTemp || 
       bestParams["top_p"] != originalTopP || 
       bestParams["max_tokens"] != originalMaxTokens {
        output.Result(fmt.Sprintf("Found improved parameters for agent %s: temp=%.2f, top_p=%.2f, max_tokens=%d",
            agent.GetID(), 
            bestParams["temperature"], 
            bestParams["top_p"], 
            bestParams["max_tokens"]))
        return bestParams, nil
    }
    
    // No improvement
    output.Info(fmt.Sprintf("No parameter improvements found for agent %s", agent.GetID()))
    return currentParams, nil
}

// Helper functions

// randomInRange generates a random float64 in the given range
func randomInRange(min, max float64) float64 {
    return min + rand.Float64()*(max-min)
}

// randomIntInRange generates a random int in the given range
func randomIntInRange(min, max int) int {
    return min + rand.Intn(max-min+1)
}

// boundedValue ensures a value is within the given range
func boundedValue(value, min, max float64) float64 {
    if value < min {
        return min
    }
    if value > max {
        return max
    }
    return value
}

// boundedIntValue ensures an int is within the given range
func boundedIntValue(value, min, max int) int {
    if value < min {
        return min
    }
    if value > max {
        return max
    }
    return value
}

// getFloat64Param gets a float64 parameter with a default value
func getFloat64Param(params map[string]interface{}, key string, defaultValue float64) float64 {
    if value, ok := params[key].(float64); ok {
        return value
    }
    return defaultValue
}

// getIntParam gets an int parameter with a default value
func getIntParam(params map[string]interface{}, key string, defaultValue int) int {
    if value, ok := params[key].(int); ok {
        return value
    }
    if value, ok := params[key].(float64); ok {
        return int(value)
    }
    return defaultValue
}

// evaluateParameters evaluates the current parameter performance
func evaluateParameters(metrics *metrics.AgentMetrics) float64 {
    // Create a composite score from various metrics
    score := 0.0
    
    // Base on success rate
    score += float64(metrics.SuccessRate) * 0.4
    
    // Quality contribution
    score += float64(metrics.AverageCompletionQuality) * 0.4
    
    // Efficiency contribution (normalize time to 0-1 scale, where 1 is best/fastest)
    avgTimeSeconds := metrics.AverageExecutionTime.Seconds()
    timeScore := 1.0 / (1.0 + avgTimeSeconds/10.0) // 10 seconds is reference point
    score += timeScore * 0.2
    
    return score
}

// simulateParameters simulates how a parameter change might affect performance
// In a real implementation, this would likely use ML to predict performance
func simulateParameters(metrics *metrics.AgentMetrics, params map[string]interface{}) float64 {
    // Extract parameters
    temperature, _ := params["temperature"].(float64)
    topP, _ := params["top_p"].(float64)
    maxTokens, _ := params["max_tokens"].(int)
    
    // Start with base score
    baseScore := evaluateParameters(metrics)
    
    // Adjust for temperature (based on common heuristics)
    tempAdjustment := 0.0
    if metrics.AverageCompletionQuality < 0.5 {
        // If quality is poor, lower temperature might help
        tempAdjustment = (0.7 - temperature) * 0.1
    } else {
        // For high quality, mild temperature is often best (around 0.7)
        tempAdjustment = -math.Abs(0.7-temperature) * 0.05
    }
    
    // Adjust for topP (usually higher is better for coherence)
    topPAdjustment := (topP - 0.5) * 0.05
    
    // Adjust for max tokens (usually more is better but with diminishing returns)
    tokenAdjustment := math.Log1p(float64(maxTokens)/1000.0) * 0.05
    
    // Apply adjustments (limited impact since this is simulation)
    return baseScore * (1.0 + tempAdjustment + topPAdjustment + tokenAdjustment)
}
```

## 2.3 Prompt Engineering System

**New Files to Create:**
- `pkg/agent/prompt/engineer.go`
- `pkg/agent/prompt/templates.go`

**Implementation Details:**

1. Create the prompt engineer interface in `engineer.go`:

```go
package prompt

import (
    "context"
    
    "github.com/theapemachine/caramba/pkg/agent/core"
    "github.com/theapemachine/caramba/pkg/agent/metrics"
)

// PromptEngineer is responsible for optimizing system prompts
type PromptEngineer interface {
    // ImprovePrompt optimizes a system prompt based on agent performance
    ImprovePrompt(ctx context.Context, agent core.Agent) (string, error)
    
    // GetPromptImprovementHistory returns the history of prompt improvements
    GetPromptImprovementHistory(agentID string) ([]PromptImprovement, error)
    
    // CreatePrompt creates a new prompt for a specific use case
    CreatePrompt(ctx context.Context, promptType string, parameters map[string]interface{}) (string, error)
}

// PromptImprovement represents a single prompt improvement attempt
type PromptImprovement struct {
    AgentID    string
    Timestamp  string
    OldPrompt  string
    NewPrompt  string
    Rationale  string
    SuccessMetricsBefore map[string]float64
    SuccessMetricsAfter  map[string]float64
}

// LLMBasedPromptEngineer implements the PromptEngineer interface using LLMs
type LLMBasedPromptEngineer struct {
    llmProvider       core.LLMProvider
    metricsCollector  metrics.MetricsCollector
    promptHistory     map[string][]PromptImprovement
    promptTemplates   map[string]string
}

// NewLLMBasedPromptEngineer creates a new LLM-based prompt engineer
func NewLLMBasedPromptEngineer(llmProvider core.LLMProvider, metricsCollector metrics.MetricsCollector) *LLMBasedPromptEngineer {
    return &LLMBasedPromptEngineer{
        llmProvider:      llmProvider,
        metricsCollector: metricsCollector,
        promptHistory:    make(map[string][]PromptImprovement),
        promptTemplates:  DefaultPromptTemplates(),
    }
}

// ImprovePrompt optimizes a system prompt based on agent performance
func (e *LLMBasedPromptEngineer) ImprovePrompt(ctx context.Context, agent core.Agent) (string, error) {
    agentID := agent.GetID()
    
    // Get the current prompt
    currentParams := agent.GetParameters()
    currentPrompt, ok := currentParams["system_prompt"].(string)
    if !ok || currentPrompt == "" {
        return "", fmt.Errorf("agent has no system prompt")
    }
    
    // Get agent metrics
    metrics, err := e.metricsCollector.GetAgentMetrics(agentID)
    if err != nil {
        return "", fmt.Errorf("failed to get agent metrics: %w", err)
    }
    
    // Prepare prompt for LLM to improve the system prompt
    improvePromptTemplate := `You are an expert prompt engineer specializing in optimizing system prompts for AI agents.

Current System Prompt:
"""
%s
"""

Agent Performance Metrics:
- Success Rate: %.2f
- Average Completion Quality: %.2f
- Total Executions: %d
- Average Execution Time: %v
- Most Used Tools: %v

Based on these metrics, please improve the system prompt to enhance the agent's performance.
Focus on:
1. Clarity and precision of instructions
2. Better guidance for tool usage
3. Optimization for the specific tasks the agent handles
4. Addressing any apparent weaknesses in the metrics

Provide only the improved system prompt without any explanation or additional text.`

    // Format the most used tools string
    toolUsageStr := ""
    for toolName, metrics := range metrics.ToolUsage {
        toolUsageStr += fmt.Sprintf("%s (%d uses, %.2f%% success), ", 
            toolName, metrics.Uses, float64(metrics.SuccessfulUses)/float64(metrics.Uses)*100)
    }
    
    // Format the prompt
    improvePrompt := fmt.Sprintf(improvePromptTemplate, 
        currentPrompt,
        metrics.SuccessRate,
        metrics.AverageCompletionQuality,
        metrics.TotalExecutions,
        metrics.AverageExecutionTime,
        toolUsageStr)
    
    // Call the LLM to improve the prompt
    response := e.llmProvider.GenerateResponse(ctx, core.LLMParams{
        Messages: []core.LLMMessage{
            {
                Role:    "user",
                Content: improvePrompt,
            },
        },
        Temperature: 0.7,
        MaxTokens:   2048,
    })
    
    if response.Error != nil {
        return "", fmt.Errorf("LLM failed to improve prompt: %w", response.Error)
    }
    
    improvedPrompt := response.Content
    
    // Verify the improved prompt isn't empty or drastically different
    if len(improvedPrompt) < 50 {
        return "", fmt.Errorf("improved prompt is too short, likely invalid")
    }
    
    // Store the improvement in history
    improvement := PromptImprovement{
        AgentID:   agentID,
        Timestamp: time.Now().Format(time.RFC3339),
        OldPrompt: currentPrompt,
        NewPrompt: improvedPrompt,
        SuccessMetricsBefore: map[string]float64{
            "SuccessRate":      float64(metrics.SuccessRate),
            "CompletionQuality": float64(metrics.AverageCompletionQuality),
        },
        // Success metrics after will be updated later
    }
    
    e.promptHistory[agentID] = append(e.promptHistory[agentID], improvement)
    
    // Get rationale for the changes (optional, separate LLM call)
    rationaleResponse := e.llmProvider.GenerateResponse(ctx, core.LLMParams{
        Messages: []core.LLMMessage{
            {
                Role: "user",
                Content: fmt.Sprintf(`
Analyze the differences between these two system prompts and explain the rationale for the changes:

Original:
"""
%s
"""

Improved:
"""
%s
"""

What specific improvements were made and how do they address the performance metrics?`, currentPrompt, improvedPrompt),
            },
        },
        Temperature: 0.7,
        MaxTokens:   1024,
    })
    
    if rationaleResponse.Error == nil {
        // If we got a rationale, store it
        improvement.Rationale = rationaleResponse.Content
    }
    
    return improvedPrompt, nil
}

// GetPromptImprovementHistory returns the history of prompt improvements
func (e *LLMBasedPromptEngineer) GetPromptImprovementHistory(agentID string) ([]PromptImprovement, error) {
    history, ok := e.promptHistory[agentID]
    if !ok {
        return nil, fmt.Errorf("no prompt improvement history for agent %s", agentID)
    }
    return history, nil
}

// CreatePrompt creates a new prompt for a specific use case
func (e *LLMBasedPromptEngineer) CreatePrompt(ctx context.Context, promptType string, parameters map[string]interface{}) (string, error) {
    // Check if we have a template for this prompt type
    templateText, ok := e.promptTemplates[promptType]
    if !ok {
        return "", fmt.Errorf("no template found for prompt type: %s", promptType)
    }
    
    // If it's a simple template with no parameters, just return it
    if len(parameters) == 0 {
        return templateText, nil
    }
    
    // Otherwise, use the LLM to fill in the template
    paramStr := ""
    for k, v := range parameters {
        paramStr += fmt.Sprintf("%s: %v\n", k, v)
    }
    
    // Call the LLM to create a tailored prompt
    response := e.llmProvider.GenerateResponse(ctx, core.LLMParams{
        Messages: []core.LLMMessage{
            {
                Role: "user",
                Content: fmt.Sprintf(`You are an expert prompt engineer.
Create a system prompt of type: %s

Template:
"""
%s
"""

Parameters to incorporate:
%s

Please provide the complete, fully formed system prompt incorporating all these parameters.
The prompt should be usable without any further modifications. Do not include any explanations.`,
                    promptType, templateText, paramStr),
            },
        },
        Temperature: 0.7,
        MaxTokens:   2048,
    })
    
    if response.Error != nil {
        return "", fmt.Errorf("LLM failed to create prompt: %w", response.Error)
    }
    
    return response.Content, nil
}
```

2. Create prompt templates in `templates.go`:

```go
package prompt

// DefaultPromptTemplates returns a map of default prompt templates
func DefaultPromptTemplates() map[string]string {
    return map[string]string{
        "researcher": `You are a web researcher that can use the browser tool to search the web.
For each research task, follow these steps:
1. Identify key aspects that need to be researched
2. Use the browser tool to search for information 
3. Navigate to relevant pages and extract content
4. Synthesize your findings into a comprehensive report
    
Your output should be formatted in Markdown with clear sections:
    
# Research Summary: [Topic]
    
## Key Findings
- Important point 1
- Important point 2
    
## Details and Analysis
[Detailed breakdown of the research]
    
## Alternative Approaches
[Compare and contrast different methods]
    
## Conclusions
[Final thoughts and recommendations]`,

        "coding_assistant": `You are a coding assistant with expertise in software development.
Your goal is to help users write high-quality, maintainable code.

When helping with code:
1. Understand the requirements thoroughly before suggesting solutions
2. Provide clear explanations for your code
3. Follow best practices for the language/framework in question
4. Consider edge cases and error handling
5. Focus on readability and maintainability

Your responses should include:
- Clean, well-documented code examples
- Explanations of key concepts and design decisions
- Potential alternatives where appropriate
- Any limitations or considerations for the suggested approach`,

        "planner": `You are an AI planning system that creates detailed step-by-step plans to achieve goals.

For any task, carefully analyze what needs to be done and create a logical sequence of steps.
Each step should specify:
1. A clear description of what needs to be accomplished
2. Which tool to use (if applicable)
3. The necessary arguments for that tool

Your plans should be comprehensive, addressing all aspects of the task.
Consider potential challenges and include steps to handle them.

Output your plan as a JSON array of steps, where each step has:
- "description": Clear explanation of what this step accomplishes
- "tool": Name of the tool to use (leave empty if no tool needed)
- "arguments": Object containing the arguments for the tool`,

        "general_assistant": `You are a helpful AI assistant. Your goal is to provide accurate, helpful responses to user requests.

When responding:
1. Answer questions directly and concisely
2. When uncertain, indicate your level of confidence
3. For complex topics, break down your explanation into digestible parts
4. Use examples when helpful
5. Format your responses for readability

You should be:
- Knowledgeable but honest about limitations
- Helpful without being verbose
- Neutral on controversial topics
- Precise in your language

For step-by-step instructions, use numbered lists.
For alternatives or options, use bullet points.
For code or technical content, use appropriate formatting.`,
    }
}
```

## 2.4 Experience Learning System

**New Files to Create:**
- `pkg/agent/learning/examples.go`
- `pkg/agent/learning/collector.go`

**Implementation Details:**

1. Define the example collection interface in `examples.go`:

```go
package learning

import (
    "context"
    "time"
    
    "github.com/theapemachine/caramba/pkg/agent/core"
)

// Example represents a single learning example
type Example struct {
    ID           string
    AgentID      string
    Prompt       string  // User input
    Response     string  // Agent response
    Tools        []ToolUse
    Quality      float32 // Quality rating (0-1)
    Source       string  // How this example was obtained
    CollectedAt  time.Time
    Tags         []string
    Metadata     map[string]interface{}
}

// ToolUse represents a tool usage within an example
type ToolUse struct {
    ToolName  string
    Args      map[string]interface{}
    Result    interface{}
    Successful bool
}

// ExampleSelector provides methods to select learning examples
type ExampleSelector interface {
    // SelectExamples selects examples matching the given criteria
    SelectExamples(ctx context.Context, criteria map[string]interface{}, limit int) ([]Example, error)
    
    // GetExampleByID retrieves a specific example by ID
    GetExampleByID(ctx context.Context, id string) (*Example, error)
    
    // CountExamples counts examples matching the given criteria
    CountExamples(ctx context.Context, criteria map[string]interface{}) (int, error)
}

// ExampleStore provides methods to store and retrieve learning examples
type ExampleStore interface {
    ExampleSelector
    
    // StoreExample stores a new learning example
    StoreExample(ctx context.Context, example *Example) error
    
    // DeleteExample deletes an example by ID
    DeleteExample(ctx context.Context, id string) error
    
    // UpdateExample updates an existing example
    UpdateExample(ctx context.Context, example *Example) error
}

// ExampleCollector collects learning examples from agent interactions
type ExampleCollector interface {
    // CollectExample collects a learning example from agent execution
    CollectExample(ctx context.Context, 
        agentID string,
        prompt core.LLMMessage, 
        response string,
        tools []ToolUse,
        quality float32) error
    
    // GetStore returns the example store
    GetStore() ExampleStore
}
```

2. Implement the collector in `collector.go`:

```go
package learning

import (
    "context"
    "fmt"
    "sync"
    "time"
    
    "github.com/google/uuid"
    "github.com/theapemachine/caramba/pkg/agent/core"
    "github.com/theapemachine/caramba/pkg/output"
)

// BasicExampleCollector implements ExampleCollector with in-memory storage
type BasicExampleCollector struct {
    store ExampleStore
}

// NewBasicExampleCollector creates a new BasicExampleCollector
func NewBasicExampleCollector(store ExampleStore) *BasicExampleCollector {
    if store == nil {
        store = NewInMemoryExampleStore()
    }
    
    return &BasicExampleCollector{
        store: store,
    }
}

// CollectExample collects a learning example from agent execution
func (c *BasicExampleCollector) CollectExample(ctx context.Context, 
    agentID string,
    prompt core.LLMMessage,
    response string,
    tools []ToolUse,
    quality float32) error {
    
    example := &Example{
        ID:           uuid.New().String(),
        AgentID:      agentID,
        Prompt:       prompt.Content,
        Response:     response,
        Tools:        tools,
        Quality:      quality,
        Source:       "agent_execution",
        CollectedAt:  time.Now(),
        Tags:         []string{},
        Metadata:     map[string]interface{}{},
    }
    
    // Add tags based on content and quality
    if quality >= 0.8 {
        example.Tags = append(example.Tags, "high_quality")
    }
    
    // Add tool usage tags
    for _, tool := range tools {
        toolTag := fmt.Sprintf("uses_%s", tool.ToolName)
        if !contains(example.Tags, toolTag) {
            example.Tags = append(example.Tags, toolTag)
        }
    }
    
    // Store the example
    err := c.store.StoreExample(ctx, example)
    if err != nil {
        return fmt.Errorf("failed to store example: %w", err)
    }
    
    output.Verbose(fmt.Sprintf("Collected learning example %s for agent %s (quality: %.2f)", example.ID, agentID, quality))
    return nil
}

// GetStore returns the example store
func (c *BasicExampleCollector) GetStore() ExampleStore {
    return c.store
}

// InMemoryExampleStore implements ExampleStore with in-memory storage
type InMemoryExampleStore struct {
    examples map[string]*Example
    mu       sync.RWMutex
}

// NewInMemoryExampleStore creates a new InMemoryExampleStore
func NewInMemoryExampleStore() *InMemoryExampleStore {
    return &InMemoryExampleStore{
        examples: make(map[string]*Example),
    }
}

// StoreExample stores a new learning example
func (s *InMemoryExampleStore) StoreExample(ctx context.Context, example *Example) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    s.examples[example.ID] = example
    return nil
}

// SelectExamples selects examples matching the given criteria
func (s *InMemoryExampleStore) SelectExamples(ctx context.Context, criteria map[string]interface{}, limit int) ([]Example, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    var result []Example
    
    for _, example := range s.examples {
        if matchesCriteria(example, criteria) {
            result = append(result, *example)
            
            if limit > 0 && len(result) >= limit {
                break
            }
        }
    }
    
    return result, nil
}

// GetExampleByID retrieves a specific example by ID
func (s *InMemoryExampleStore) GetExampleByID(ctx context.Context, id string) (*Example, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    example, exists := s.examples[id]
    if !exists {
        return nil, fmt.Errorf("example not found: %s", id)
    }
    
    return example, nil
}

// CountExamples counts examples matching the given criteria
func (s *InMemoryExampleStore) CountExamples(ctx context.Context, criteria map[string]interface{}) (int, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    count := 0
    for _, example := range s.examples {
        if matchesCriteria(example, criteria) {
            count++
        }
    }
    
    return count, nil
}

// DeleteExample deletes an example by ID
func (s *InMemoryExampleStore) DeleteExample(ctx context.Context, id string) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    if _, exists := s.examples[id]; !exists {
        return fmt.Errorf("example not found: %s", id)
    }
    
    delete(s.examples, id)
    return nil
}

// UpdateExample updates an existing example
func (s *InMemoryExampleStore) UpdateExample(ctx context.Context, example *Example) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    if _, exists := s.examples[example.ID]; !exists {
        return fmt.Errorf("example not found: %s", example.ID)
    }
    
    s.examples[example.ID] = example
    return nil
}

// Helper functions

// contains checks if a string slice contains a value
func contains(slice []string, value string) bool {
    for _, item := range slice {
        if item == value {
            return true
        }
    }
    return false
}

// matchesCriteria checks if an example matches the given criteria
func matchesCriteria(example *Example, criteria map[string]interface{}) bool {
    for key, value := range criteria {
        switch key {
        case "agent_id":
            if strVal, ok := value.(string); ok && example.AgentID != strVal {
                return false
            }
        case "min_quality":
            if floatVal, ok := value.(float32); ok && example.Quality < floatVal {
                return false
            }
        case "max_quality":
            if floatVal, ok := value.(float32); ok && example.Quality > floatVal {
                return false
            }
        case "source":
            if strVal, ok := value.(string); ok && example.Source != strVal {
                return false
            }
        case "tag":
            if strVal, ok := value.(string); ok && !contains(example.Tags, strVal) {
                return false
            }
        case "tool":
            if strVal, ok := value.(string); ok {
                found := false
                for _, tool := range example.Tools {
                    if tool.ToolName == strVal {
                        found = true
                        break
                    }
                }
                if !found {
                    return false
                }
            }
        case "since":
            if timeVal, ok := value.(time.Time); ok && example.CollectedAt.Before(timeVal) {
                return false
            }
        case "until":
            if timeVal, ok := value.(time.Time); ok && example.CollectedAt.After(timeVal) {
                return false
            }
        }
    }
    
    return true
}
```

# Agent Self-Improvement Implementation

## Integration with Base Agent

To make self-improvement a first-class feature of the Caramba framework, we need to modify the `BaseAgent` struct in `pkg/agent/core/base_agent.go` to include these new capabilities:

```go
// BaseAgent provides a base implementation of the Agent interface
type BaseAgent struct {
    Name           string
    LLM            LLMProvider
    Memory         Memory
    Params         LLMParams
    Tools          []Tool
    Planner        Planner
    Messenger      Messenger
    Metrics        metrics.MetricsCollector    // New: metrics collector
    Optimizer      optimization.Optimizer      // New: parameter optimizer
    PromptEngineer prompt.PromptEngineer       // New: prompt engineer
    ExampleCollector learning.ExampleCollector // New: learning examples collector
    toolsMu        sync.RWMutex
    IterationLimit int
    LastOptimized  time.Time                   // New: track last optimization time
}
```

### Self-Improvement Command

Let's add a new command to trigger self-improvement in `pkg/agent/cmd/improve.go`:

```go
package cmd

import (
    "fmt"
    "os"
    "time"

    "github.com/spf13/cobra"
    "github.com/theapemachine/caramba/pkg/agent/core"
    "github.com/theapemachine/caramba/pkg/agent/metrics"
    "github.com/theapemachine/caramba/pkg/agent/optimization"
    "github.com/theapemachine/caramba/pkg/agent/prompt"
    "github.com/theapemachine/caramba/pkg/output"
    "github.com/theapemachine/errnie"
)

/*
improveCmd is a command that improves an agent based on performance metrics
*/
var improveCmd = &cobra.Command{
    Use:   "improve [agent_id]",
    Short: "Improve agent performance",
    Long:  `Analyzes agent performance metrics and applies optimizations to improve future performance`,
    Args:  cobra.MinimumNArgs(1),
    Run: func(cmd *cobra.Command, args []string) {
        os.Setenv("LOG_LEVEL", "debug")
        os.Setenv("LOGFILE", "true")
        errnie.InitLogger()

        // Get the agent ID from args
        agentID := args[0]

        // Get API key from flags or environment
        apiKey, _ := cmd.Flags().GetString("api-key")
        if apiKey == "" {
            apiKey = os.Getenv("OPENAI_API_KEY")
        }

        if apiKey == "" {
            errnie.Error(fmt.Errorf("API key not provided. Use --api-key flag or set OPENAI_API_KEY environment variable"))
            return
        }

        // Create metrics collector
        metricsCollector := metrics.NewInMemoryMetricsCollector()

        // Load agent state from repository
        // (In a real implementation, you'd load the agent from storage)
        agent, err := loadAgent(agentID, apiKey, metricsCollector)
        if err != nil {
            errnie.Error(fmt.Errorf("failed to load agent: %w", err))
            return
        }

        // Get improvement type from flags
        improveType, _ := cmd.Flags().GetString("type")

        // Create optimizers
        optimizer := optimization.NewBasicOptimizer(metricsCollector)
        promptEngineer := prompt.NewLLMBasedPromptEngineer(agent.LLM, metricsCollector)

        // Perform the improvement
        switch improveType {
        case "all":
            output.Title("COMPREHENSIVE AGENT IMPROVEMENT")
            performAllImprovements(cmd.Context(), agent, optimizer, promptEngineer)
        case "parameters":
            output.Title("PARAMETER OPTIMIZATION")
            optimizeParameters(cmd.Context(), agent, optimizer)
        case "prompt":
            output.Title("PROMPT ENGINEERING")
            improvePrompt(cmd.Context(), agent, promptEngineer)
        default:
            errnie.Error(fmt.Errorf("unknown improvement type: %s", improveType))
            return
        }

        // Save the improved agent
        if err := saveAgent(agent); err != nil {
            errnie.Error(fmt.Errorf("failed to save agent: %w", err))
            return
        }

        output.Result(fmt.Sprintf("Agent %s successfully improved", agentID))
    },
}

func init() {
    rootCmd.AddCommand(improveCmd)
    improveCmd.Flags().String("api-key", "", "API key for the LLM provider (or set OPENAI_API_KEY env var)")
    improveCmd.Flags().String("type", "all", "Type of improvement to apply (all, parameters, prompt)")
}

// performAllImprovements performs all types of improvements
func performAllImprovements(ctx context.Context, agent core.Agent, optimizer optimization.Optimizer, promptEngineer prompt.PromptEngineer) {
    // First optimize parameters
    output.Stage(1, "Parameter Optimization")
    optimizeParameters(ctx, agent, optimizer)

    // Then improve prompt
    output.Stage(2, "Prompt Engineering")
    improvePrompt(ctx, agent, promptEngineer)
}

// optimizeParameters performs parameter optimization
func optimizeParameters(ctx context.Context, agent core.Agent, optimizer optimization.Optimizer) {
    output.Info(fmt.Sprintf("Starting parameter optimization for agent %s", agent.GetID()))
    
    optSpinner := output.StartSpinner("Analyzing metrics and optimizing parameters")
    err := optimizer.OptimizeAgent(ctx, agent)
    
    if err != nil {
        output.StopSpinner(optSpinner, "")
        output.Error("Parameter optimization failed", err)
        return
    }
    
    output.StopSpinner(optSpinner, "Parameter optimization completed")
    
    // Get the optimization history
    history, err := optimizer.GetOptimizationHistory(agent.GetID())
    if err == nil && len(history) > 0 {
        // Show the latest optimization
        latest := history[len(history)-1]
        
        output.Info("Parameter changes:")
        for param, oldVal := range latest.OldParams {
            if newVal, ok := latest.NewParams[param]; ok && oldVal != newVal {
                output.Info(fmt.Sprintf("  - %s: %v → %v", param, oldVal, newVal))
            }
        }
    }
}

// improvePrompt performs prompt engineering
func improvePrompt(ctx context.Context, agent core.Agent, promptEngineer prompt.PromptEngineer) {
    output.Info(fmt.Sprintf("Starting prompt engineering for agent %s", agent.GetID()))
    
    promptSpinner := output.StartSpinner("Analyzing metrics and improving prompt")
    improvedPrompt, err := promptEngineer.ImprovePrompt(ctx, agent)
    
    if err != nil {
        output.StopSpinner(promptSpinner, "")
        output.Error("Prompt improvement failed", err)
        return
    }
    
    output.StopSpinner(promptSpinner, "Prompt improvement completed")
    
    // Extract current parameters
    currentParams := agent.GetParameters()
    currentPrompt, _ := currentParams["system_prompt"].(string)
    
    // Apply the new prompt
    newParams := make(map[string]interface{})
    for k, v := range currentParams {
        newParams[k] = v
    }
    newParams["system_prompt"] = improvedPrompt
    
    if err := agent.SetParameters(newParams); err != nil {
        output.Error("Failed to apply new prompt", err)
        return
    }
    
    // Show a summary of changes
    promptDiff := summarizePromptChanges(currentPrompt, improvedPrompt)
    output.Info("Prompt improvements:")
    for _, change := range promptDiff {
        output.Info(fmt.Sprintf("  - %s", change))
    }
}

// loadAgent loads an agent by ID (placeholder implementation)
func loadAgent(agentID string, apiKey string, metricsCollector metrics.MetricsCollector) (core.Agent, error) {
    // In a real implementation, you'd load the agent from storage
    
    // For demonstration, we'll create a new agent
    agent := core.NewBaseAgent(agentID)
    
    // Set up the agent
    agent.SetLLM(llm.NewOpenAIProvider(apiKey, "gpt-4o-mini"))
    agent.SetMemory(memory.NewInMemoryStore())
    agent.SetIterationLimit(3)
    
    // Set metrics collector
    agent.Metrics = metricsCollector
    
    // Set a default system prompt
    agent.SetSystemPrompt(`You are a helpful AI assistant. Your goal is to provide accurate, helpful responses to user requests.`)
    
    return agent, nil
}

// saveAgent saves an agent (placeholder implementation)
func saveAgent(agent core.Agent) error {
    // In a real implementation, you'd save the agent to storage
    output.Verbose(fmt.Sprintf("Agent %s parameters saved", agent.GetID()))
    return nil
}

// summarizePromptChanges summarizes changes between prompts
func summarizePromptChanges(oldPrompt, newPrompt string) []string {
    // This is a simplified implementation
    // In a real system, you'd use a diff algorithm
    
    // For now, just mention that the prompt was updated
    return []string{
        "Updated system prompt with improved instructions",
        fmt.Sprintf("Length changed from %d to %d characters", len(oldPrompt), len(newPrompt)),
    }
}
```

### Agent Self-Improvement API Extension

Let's add methods to the `Agent` interface in `pkg/agent/core/interfaces.go` to enable self-improvement:

```go
// Agent represents the interface for an LLM-powered agent.
type Agent interface {
    // ... existing methods ...
    
    // GetID returns the agent's unique identifier
    GetID() string
    
    // GetParameters returns the agent's current parameters
    GetParameters() map[string]interface{}
    
    // SetParameters sets the agent's parameters
    SetParameters(params map[string]interface{}) error
    
    // Improve triggers the agent's self-improvement mechanism
    Improve(ctx context.Context) error
    
    // GetMetrics returns the agent's metrics
    GetMetrics(ctx context.Context) (*metrics.AgentMetrics, error)
}
```

### BaseAgent Implementation of Self-Improvement API

Now, let's implement these methods in the `BaseAgent` struct:

```go
// GetID returns the agent's unique identifier
func (a *BaseAgent) GetID() string {
    return a.Name
}

// GetParameters returns the agent's current parameters
func (a *BaseAgent) GetParameters() map[string]interface{} {
    // Create a copy of the parameters to avoid external modification
    params := make(map[string]interface{})
    
    // Add LLM parameters
    params["temperature"] = a.Params.Temperature
    params["top_p"] = a.Params.TopP
    params["max_tokens"] = a.Params.MaxTokens
    params["frequency_penalty"] = a.Params.FrequencyPenalty
    params["presence_penalty"] = a.Params.PresencePenalty
    
    // Add system prompt if available
    if len(a.Params.Messages) > 0 && a.Params.Messages[0].Role == "system" {
        params["system_prompt"] = a.Params.Messages[0].Content
    }
    
    return params
}

// SetParameters sets the agent's parameters
func (a *BaseAgent) SetParameters(params map[string]interface{}) error {
    // Update LLM parameters
    if temp, ok := params["temperature"].(float64); ok {
        a.Params.Temperature = temp
    }
    
    if topP, ok := params["top_p"].(float64); ok {
        a.Params.TopP = topP
    }
    
    if maxTokens, ok := params["max_tokens"].(int); ok {
        a.Params.MaxTokens = maxTokens
    }
    
    if freqPenalty, ok := params["frequency_penalty"].(float64); ok {
        a.Params.FrequencyPenalty = freqPenalty
    }
    
    if presPenalty, ok := params["presence_penalty"].(float64); ok {
        a.Params.PresencePenalty = presPenalty
    }
    
    // Update system prompt if provided
    if systemPrompt, ok := params["system_prompt"].(string); ok {
        // Remove any existing system message
        var nonSystemMessages []core.LLMMessage
        for _, msg := range a.Params.Messages {
            if msg.Role != "system" {
                nonSystemMessages = append(nonSystemMessages, msg)
            }
        }
        
        // Add the new system message at the beginning
        a.Params.Messages = append([]core.LLMMessage{
            {
                Role:    "system",
                Content: systemPrompt,
            },
        }, nonSystemMessages...)
    }
    
    return nil
}

// Improve triggers the agent's self-improvement mechanism
func (a *BaseAgent) Improve(ctx context.Context) error {
    if a.Optimizer == nil && a.PromptEngineer == nil {
        return fmt.Errorf("no optimization components available")
    }
    
    // Check if enough time has passed since last optimization
    minInterval := 24 * time.Hour // Default: daily
    if time.Since(a.LastOptimized) < minInterval {
        return fmt.Errorf("optimization cooldown period active (last run: %v)", a.LastOptimized)
    }
    
    output.Info(fmt.Sprintf("Starting self-improvement for agent %s", a.Name))
    
    // Step 1: Parameter optimization
    if a.Optimizer != nil {
        err := a.Optimizer.OptimizeAgent(ctx, a)
        if err != nil {
            output.Warn(fmt.Sprintf("Parameter optimization warning: %v", err))
        } else {
            output.Result("Parameter optimization completed")
        }
    }
    
    // Step 2: Prompt engineering
    if a.PromptEngineer != nil {
        improvedPrompt, err := a.PromptEngineer.ImprovePrompt(ctx, a)
        if err != nil {
            output.Warn(fmt.Sprintf("Prompt improvement warning: %v", err))
        } else {
            // Apply the improved prompt
            params := a.GetParameters()
            params["system_prompt"] = improvedPrompt
            if err := a.SetParameters(params); err != nil {
                output.Warn(fmt.Sprintf("Failed to apply improved prompt: %v", err))
            } else {
                output.Result("Prompt improvement completed")
            }
        }
    }
    
    // Update last optimized timestamp
    a.LastOptimized = time.Now()
    
    output.Result(fmt.Sprintf("Self-improvement completed for agent %s", a.Name))
    return nil
}

// GetMetrics returns the agent's metrics
func (a *BaseAgent) GetMetrics(ctx context.Context) (*metrics.AgentMetrics, error) {
    if a.Metrics == nil {
        return nil, fmt.Errorf("no metrics collector available")
    }
    
    return a.Metrics.GetAgentMetrics(a.Name)
}
```

### Auto-Improvement Trigger in Execute Method

Let's add code to automatically trigger self-improvement after a certain number of executions by modifying the `Execute` method:

```go
// Execute runs the agent with the provided input and returns a response
func (a *BaseAgent) Execute(ctx context.Context, message core.LLMMessage) (string, error) {
    startTime := time.Now()
    
    // ... existing execution code ...

    // Check if it's time for self-improvement
    if a.Metrics != nil {
        // Get current metrics
        metrics, err := a.Metrics.GetAgentMetrics(a.Name)
        if err == nil && metrics.TotalExecutions > 0 && metrics.TotalExecutions % 50 == 0 {
            // After every 50 executions, consider self-improvement
            if time.Since(a.LastOptimized) > 24*time.Hour {
                output.Info("Scheduled self-improvement triggered")
                go func() {
                    // Run improvement in background
                    improvementCtx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
                    defer cancel()
                    
                    if err := a.Improve(improvementCtx); err != nil {
                        output.Warn(fmt.Sprintf("Scheduled self-improvement failed: %v", err))
                    }
                }()
            }
        }
    }

    return response.String(), nil
}
```

### AgentBuilder Extension for Self-Improvement

Finally, let's extend the `AgentBuilder` to include self-improvement capabilities:

```go
// WithMetricsCollector sets the metrics collector for the agent
func (b *AgentBuilder) WithMetricsCollector(metrics metrics.MetricsCollector) *AgentBuilder {
    b.agent.Metrics = metrics
    return b
}

// WithOptimizer sets the parameter optimizer for the agent
func (b *AgentBuilder) WithOptimizer(optimizer optimization.Optimizer) *AgentBuilder {
    b.agent.Optimizer = optimizer
    return b
}

// WithPromptEngineer sets the prompt engineer for the agent
func (b *AgentBuilder) WithPromptEngineer(promptEngineer prompt.PromptEngineer) *AgentBuilder {
    b.agent.PromptEngineer = promptEngineer
    return b
}

// WithExampleCollector sets the learning example collector for the agent
func (b *AgentBuilder) WithExampleCollector(collector learning.ExampleCollector) *AgentBuilder {
    b.agent.ExampleCollector = collector
    return b
}
```

## Self-Improvement Workflow

With these components in place, here's how the self-improvement process works:

1. **Metrics Collection**: During regular agent execution, the system collects metrics on:
   - Tool usage success rates
   - Execution times
   - Token consumption
   - Memory operations
   - Completion quality (when explicitly rated)

2. **Parameter Optimization**: The optimizer periodically:
   - Analyzes collected metrics
   - Tries different parameter configurations
   - Selects the best performing parameters (temperature, top_p, etc.)
   - Updates the agent's configuration

3. **Prompt Engineering**: The prompt engineer:
   - Analyzes the agent's performance patterns
   - Identifies weaknesses in the current system prompt
   - Generates an improved prompt that addresses those issues
   - Updates the agent's system prompt

4. **Learning Example Collection**: The example collector:
   - Captures successful interactions (high-quality completions)
   - Stores them with metadata about tools used and performance metrics
   - Makes these examples available for future reference and optimization

5. **Triggering**: Self-improvement can be triggered:
   - Manually via the `improve` command
   - Automatically after a certain number of executions
   - Based on performance degradation detection

This self-improvement cycle creates a feedback loop that continuously enhances agent performance over time.

## Future Enhancements

Several enhancements could be added to this system:

1. **A/B Testing**: Implement a mechanism to compare different agent configurations side-by-side
2. **Reinforcement Learning**: Add support for more sophisticated RL-based parameter optimization
3. **Cross-Agent Learning**: Allow agents to learn from each other's successes and failures
4. **Human Feedback Integration**: Incorporate explicit human feedback into the optimization process
5. **Automatic Mode Switching**: Detect the most appropriate agent mode based on task characteristics

These components provide a solid foundation for building self-improving agents in the Caramba framework.