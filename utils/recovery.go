package utils

import (
	"context"
	"sync"
)

// HealthStatus represents the current health state of a component
type HealthStatus int

const (
	StatusHealthy HealthStatus = iota
	StatusDegraded
	StatusUnhealthy
)

// HealthMonitor interface defines methods for monitoring component health
type HealthMonitor interface {
	CheckHealth() HealthStatus
	GetMetrics() map[string]interface{}
}

// CleanupHandler defines the interface for cleanup operations
type CleanupHandler interface {
	Cleanup(ctx context.Context) error
	Priority() int
}

// RecoveryStrategy defines how to handle different types of errors
type RecoveryStrategy interface {
	HandleError(ctx context.Context, err *ErrorWithContext) error
	Cleanup(ctx context.Context) error
	Priority() int
}

// RecoveryManager handles error recovery and system health
type RecoveryManager struct {
	strategies      map[ErrorSeverity][]RecoveryStrategy
	monitors        map[string]HealthMonitor
	cleanupHandlers []CleanupHandler
	mu              sync.RWMutex
}

// NewRecoveryManager creates a new RecoveryManager
func NewRecoveryManager() *RecoveryManager {
	return &RecoveryManager{
		strategies:      make(map[ErrorSeverity][]RecoveryStrategy),
		monitors:        make(map[string]HealthMonitor),
		cleanupHandlers: make([]CleanupHandler, 0),
	}
}

// AddStrategy adds a recovery strategy for a specific error severity
func (rm *RecoveryManager) AddStrategy(severity ErrorSeverity, strategy RecoveryStrategy) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	if _, exists := rm.strategies[severity]; !exists {
		rm.strategies[severity] = make([]RecoveryStrategy, 0)
	}
	rm.strategies[severity] = append(rm.strategies[severity], strategy)

	// Sort strategies by priority
	strategies := rm.strategies[severity]
	sortStrategiesByPriority(strategies)
}

// AddMonitor adds a health monitor for a component
func (rm *RecoveryManager) AddMonitor(name string, monitor HealthMonitor) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	rm.monitors[name] = monitor
}

// AddCleanupHandler adds a cleanup handler
func (rm *RecoveryManager) AddCleanupHandler(handler CleanupHandler) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	rm.cleanupHandlers = append(rm.cleanupHandlers, handler)
	sortCleanupHandlersByPriority(rm.cleanupHandlers)
}

// HandleError processes an error using appropriate recovery strategies
func (rm *RecoveryManager) HandleError(ctx context.Context, err *ErrorWithContext) error {
	rm.mu.RLock()
	strategies := rm.strategies[err.Severity]
	rm.mu.RUnlock()

	for _, strategy := range strategies {
		if err := strategy.HandleError(ctx, err); err != nil {
			continue // Try next strategy if current one fails
		}
		return nil // Return on first successful strategy
	}

	return err.Err // Return original error if no strategy succeeded
}

// Cleanup performs cleanup operations in priority order
func (rm *RecoveryManager) Cleanup(ctx context.Context) error {
	rm.mu.RLock()
	handlers := make([]CleanupHandler, len(rm.cleanupHandlers))
	copy(handlers, rm.cleanupHandlers)
	rm.mu.RUnlock()

	for _, handler := range handlers {
		if err := handler.Cleanup(ctx); err != nil {
			return err
		}
	}
	return nil
}

// GetHealthStatus returns the overall system health status
func (rm *RecoveryManager) GetHealthStatus() map[string]HealthStatus {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	status := make(map[string]HealthStatus)
	for name, monitor := range rm.monitors {
		status[name] = monitor.CheckHealth()
	}
	return status
}

// Helper function to sort strategies by priority
func sortStrategiesByPriority(strategies []RecoveryStrategy) {
	for i := 0; i < len(strategies)-1; i++ {
		for j := i + 1; j < len(strategies); j++ {
			if strategies[i].Priority() < strategies[j].Priority() {
				strategies[i], strategies[j] = strategies[j], strategies[i]
			}
		}
	}
}

// Helper function to sort cleanup handlers by priority
func sortCleanupHandlersByPriority(handlers []CleanupHandler) {
	for i := 0; i < len(handlers)-1; i++ {
		for j := i + 1; j < len(handlers); j++ {
			if handlers[i].Priority() < handlers[j].Priority() {
				handlers[i], handlers[j] = handlers[j], handlers[i]
			}
		}
	}
}
