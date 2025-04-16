package registry

import (
	"context"
	"fmt"
	"sync"
)

type MockRegistry struct {
	mu    sync.RWMutex
	store map[string]interface{}
}

func NewMockRegistry() *MockRegistry {
	return &MockRegistry{store: make(map[string]interface{})}
}

func (m *MockRegistry) Get(ctx context.Context, key string, collector any) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	v, ok := m.store[key]
	if !ok {
		return fmt.Errorf("not found")
	}
	switch c := collector.(type) {
	case *string:
		*c = v.(string)
	case *map[string]any:
		*c = v.(map[string]any)
	default:
		return fmt.Errorf("unsupported collector type")
	}
	return nil
}

func (m *MockRegistry) Register(ctx context.Context, key string, value any) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.store[key] = value
	return nil
}

func (m *MockRegistry) Unregister(ctx context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.store, key)
	return nil
}
