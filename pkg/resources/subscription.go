package resources

import (
	"context"
	"sync"
	"time"
)

// Subscription represents a subscription to a resource
type Subscription struct {
	URI       string
	Channel   chan ResourceContent
	CreatedAt time.Time
	Context   context.Context
	Cancel    context.CancelFunc
}

// SubscriptionManager handles resource subscriptions
type SubscriptionManager struct {
	subscriptions map[string][]*Subscription
	mu            sync.RWMutex
}

// NewSubscriptionManager creates a new SubscriptionManager
func NewSubscriptionManager() *SubscriptionManager {
	return &SubscriptionManager{
		subscriptions: make(map[string][]*Subscription),
	}
}

// Subscribe subscribes to a resource
func (m *SubscriptionManager) Subscribe(ctx context.Context, uri string) (*Subscription, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Create a new subscription
	subCtx, cancel := context.WithCancel(ctx)
	sub := &Subscription{
		URI:       uri,
		Channel:   make(chan ResourceContent, 10),
		CreatedAt: time.Now(),
		Context:   subCtx,
		Cancel:    cancel,
	}

	// Add the subscription to the manager
	m.subscriptions[uri] = append(m.subscriptions[uri], sub)

	// Start a goroutine to clean up the subscription when the context is done
	go func() {
		<-subCtx.Done()
		m.Unsubscribe(uri, sub)
	}()

	return sub, nil
}

// Unsubscribe unsubscribes from a resource
func (m *SubscriptionManager) Unsubscribe(uri string, sub *Subscription) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Find and remove the subscription
	if subs, ok := m.subscriptions[uri]; ok {
		for i, s := range subs {
			if s == sub {
				// Close the channel
				close(s.Channel)
				// Remove the subscription
				m.subscriptions[uri] = append(subs[:i], subs[i+1:]...)
				// If no more subscriptions for this URI, remove the URI entry
				if len(m.subscriptions[uri]) == 0 {
					delete(m.subscriptions, uri)
				}
				break
			}
		}
	}
}

// Notify notifies all subscribers of a resource update
func (m *SubscriptionManager) Notify(uri string, content ResourceContent) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if subs, ok := m.subscriptions[uri]; ok {
		for _, sub := range subs {
			select {
			case sub.Channel <- content:
				// Content sent successfully
			default:
				// Channel is full, skip this notification
			}
		}
	}
}

// GetSubscriptions returns all subscriptions for a resource
func (m *SubscriptionManager) GetSubscriptions(uri string) []*Subscription {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if subs, ok := m.subscriptions[uri]; ok {
		return subs
	}
	return nil
}

// GetSubscriptionCount returns the number of subscriptions for a resource
func (m *SubscriptionManager) GetSubscriptionCount(uri string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if subs, ok := m.subscriptions[uri]; ok {
		return len(subs)
	}
	return 0
}

// Close closes all subscriptions
func (m *SubscriptionManager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for uri, subs := range m.subscriptions {
		for _, sub := range subs {
			sub.Cancel()
			close(sub.Channel)
		}
		delete(m.subscriptions, uri)
	}
}
