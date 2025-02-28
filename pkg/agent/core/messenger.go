package core

import (
	"context"
	"errors"
	"math/rand"
	"sync"
	"time"
)

// MessageType constants
const (
	MessageTypeText    = "text"
	MessageTypeData    = "data"
	MessageTypeCommand = "command"
	MessageTypeSystem  = "system"
)

// Message represents a communication between agents
type Message struct {
	ID        string                 // Unique identifier
	Sender    string                 // ID of the sending agent
	Recipient string                 // ID of the recipient agent (empty for topics)
	Topic     string                 // Topic name (empty for direct messages)
	Content   string                 // Message content
	Timestamp time.Time              // When the message was sent
	Type      string                 // Message type (text, data, command, etc.)
	Metadata  map[string]interface{} // Additional metadata
}

// NewMessage creates a new message
func NewMessage(sender, recipient, topic, content, messageType string, metadata map[string]interface{}) *Message {
	if messageType == "" {
		messageType = MessageTypeText
	}

	if metadata == nil {
		metadata = make(map[string]interface{})
	}

	return &Message{
		ID:        generateMessageID(),
		Sender:    sender,
		Recipient: recipient,
		Topic:     topic,
		Content:   content,
		Timestamp: time.Now(),
		Type:      messageType,
		Metadata:  metadata,
	}
}

// IsDirect returns true if this is a direct message
func (m *Message) IsDirect() bool {
	return m.Recipient != "" && m.Topic == ""
}

// IsTopic returns true if this is a topic message
func (m *Message) IsTopic() bool {
	return m.Topic != "" && m.Recipient == ""
}

// IsBroadcast returns true if this is a broadcast message
func (m *Message) IsBroadcast() bool {
	return m.Recipient == "" && m.Topic == ""
}

// Topic represents a channel for publish/subscribe communication
type Topic struct {
	Name        string    // Topic name
	Description string    // Topic description
	Subscribers []string  // IDs of subscribed agents
	Creator     string    // ID of the agent that created the topic
	CreatedAt   time.Time // When the topic was created
}

// NewTopic creates a new topic
func NewTopic(name, description, creator string) *Topic {
	return &Topic{
		Name:        name,
		Description: description,
		Subscribers: []string{creator}, // Creator is automatically subscribed
		Creator:     creator,
		CreatedAt:   time.Now(),
	}
}

// HasSubscriber checks if an agent is subscribed to this topic
func (t *Topic) HasSubscriber(agentID string) bool {
	for _, subscriber := range t.Subscribers {
		if subscriber == agentID {
			return true
		}
	}
	return false
}

// AddSubscriber adds an agent as a subscriber
func (t *Topic) AddSubscriber(agentID string) bool {
	if t.HasSubscriber(agentID) {
		return false // Already subscribed
	}

	t.Subscribers = append(t.Subscribers, agentID)
	return true
}

// RemoveSubscriber removes an agent from subscribers
func (t *Topic) RemoveSubscriber(agentID string) bool {
	for i, subscriber := range t.Subscribers {
		if subscriber == agentID {
			// Remove the subscriber
			t.Subscribers = append(t.Subscribers[:i], t.Subscribers[i+1:]...)
			return true
		}
	}
	return false // Not subscribed
}

// Messenger interface for agent communication
type Messenger interface {
	// Send a direct message to another agent
	SendDirect(ctx context.Context, to string, content string, messageType string, metadata map[string]interface{}) (string, error)

	// Publish a message to a topic
	Publish(ctx context.Context, topic string, content string, messageType string, metadata map[string]interface{}) (string, error)

	// Broadcast a message to all agents
	Broadcast(ctx context.Context, content string, messageType string, metadata map[string]interface{}) ([]string, error)

	// Subscribe to a topic
	Subscribe(ctx context.Context, topic string) error

	// Unsubscribe from a topic
	Unsubscribe(ctx context.Context, topic string) error

	// Create a new topic
	CreateTopic(ctx context.Context, name string, description string) error

	// Get messages for an agent (both direct and from subscribed topics)
	GetMessages(ctx context.Context, since time.Time) ([]Message, error)

	// Get list of available topics
	GetTopics(ctx context.Context) ([]Topic, error)

	// Get the agent ID associated with this messenger
	GetAgentID() string
}

// MessageRegistry is a central registry for all messengers and topics
type MessageRegistry struct {
	messengers map[string]Messenger
	topics     map[string]*Topic
	messages   []Message
	mutex      sync.RWMutex
}

// Global message registry
var globalRegistry = NewMessageRegistry()

// NewMessageRegistry creates a new message registry
func NewMessageRegistry() *MessageRegistry {
	return &MessageRegistry{
		messengers: make(map[string]Messenger),
		topics:     make(map[string]*Topic),
		messages:   make([]Message, 0),
	}
}

// RegisterMessenger adds a messenger to the registry
func (r *MessageRegistry) RegisterMessenger(agentID string, messenger Messenger) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	r.messengers[agentID] = messenger
}

// UnregisterMessenger removes a messenger from the registry
func (r *MessageRegistry) UnregisterMessenger(agentID string) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	delete(r.messengers, agentID)

	// Also remove agent from all topics
	for _, topic := range r.topics {
		topic.RemoveSubscriber(agentID)
	}
}

// AddTopic adds a new topic to the registry
func (r *MessageRegistry) AddTopic(topic *Topic) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if _, exists := r.topics[topic.Name]; exists {
		return errors.New("topic already exists")
	}

	r.topics[topic.Name] = topic
	return nil
}

// GetTopic retrieves a topic from the registry
func (r *MessageRegistry) GetTopic(name string) (*Topic, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	topic, exists := r.topics[name]
	if !exists {
		return nil, errors.New("topic not found")
	}

	return topic, nil
}

// GetAllTopics returns all topics in the registry
func (r *MessageRegistry) GetAllTopics() []*Topic {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	topics := make([]*Topic, 0, len(r.topics))
	for _, topic := range r.topics {
		topics = append(topics, topic)
	}

	return topics
}

// AddMessage adds a message to the registry
func (r *MessageRegistry) AddMessage(msg Message) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	r.messages = append(r.messages, msg)

	// Notify recipient if it's a direct message
	if msg.IsDirect() {
		if messenger, exists := r.messengers[msg.Recipient]; exists {
			go func(m Message) {
				// This is a simplification - in a real implementation,
				// we'd need to handle the response or error
				if agent, ok := messenger.(interface {
					HandleMessage(context.Context, Message) (string, error)
				}); ok {
					_, _ = agent.HandleMessage(context.Background(), m)
				}
			}(msg)
		}
	}

	// Notify subscribers if it's a topic message
	if msg.IsTopic() {
		if topic, exists := r.topics[msg.Topic]; exists {
			for _, subscriberID := range topic.Subscribers {
				if messenger, exists := r.messengers[subscriberID]; exists {
					go func(m Message, id string) {
						if id != m.Sender { // Don't send back to sender
							if agent, ok := messenger.(interface {
								HandleMessage(context.Context, Message) (string, error)
							}); ok {
								_, _ = agent.HandleMessage(context.Background(), m)
							}
						}
					}(msg, subscriberID)
				}
			}
		}
	}

	// Notify everyone if it's a broadcast
	if msg.IsBroadcast() {
		for agentID, messenger := range r.messengers {
			if agentID != msg.Sender { // Don't send back to sender
				go func(m Message, id string, msgr Messenger) {
					if agent, ok := msgr.(interface {
						HandleMessage(context.Context, Message) (string, error)
					}); ok {
						_, _ = agent.HandleMessage(context.Background(), m)
					}
				}(msg, agentID, messenger)
			}
		}
	}
}

// GetMessagesForAgent retrieves messages for a specific agent
func (r *MessageRegistry) GetMessagesForAgent(agentID string, since time.Time) []Message {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	var results []Message

	// Find subscribed topics for this agent
	subscribedTopics := make(map[string]bool)
	for topicName, topic := range r.topics {
		if topic.HasSubscriber(agentID) {
			subscribedTopics[topicName] = true
		}
	}

	// Filter messages
	for _, msg := range r.messages {
		if msg.Timestamp.Before(since) {
			continue
		}

		// Direct messages to this agent
		if msg.IsDirect() && msg.Recipient == agentID {
			results = append(results, msg)
			continue
		}

		// Topic messages for subscribed topics
		if msg.IsTopic() && subscribedTopics[msg.Topic] {
			results = append(results, msg)
			continue
		}

		// Broadcast messages
		if msg.IsBroadcast() {
			results = append(results, msg)
		}
	}

	return results
}

// InMemoryMessenger provides an in-memory implementation of the Messenger interface
type InMemoryMessenger struct {
	agentID string
}

// NewInMemoryMessenger creates a new in-memory messenger
func NewInMemoryMessenger(agentID string) *InMemoryMessenger {
	messenger := &InMemoryMessenger{
		agentID: agentID,
	}

	// Register with global registry
	globalRegistry.RegisterMessenger(agentID, messenger)

	return messenger
}

// SendDirect sends a direct message to another agent
func (m *InMemoryMessenger) SendDirect(ctx context.Context, to string, content string, messageType string, metadata map[string]interface{}) (string, error) {
	msg := NewMessage(m.agentID, to, "", content, messageType, metadata)
	globalRegistry.AddMessage(*msg)
	return msg.ID, nil
}

// Publish publishes a message to a topic
func (m *InMemoryMessenger) Publish(ctx context.Context, topic string, content string, messageType string, metadata map[string]interface{}) (string, error) {
	// Check if topic exists
	if _, err := globalRegistry.GetTopic(topic); err != nil {
		return "", errors.New("topic does not exist")
	}

	msg := NewMessage(m.agentID, "", topic, content, messageType, metadata)
	globalRegistry.AddMessage(*msg)
	return msg.ID, nil
}

// Broadcast sends a message to all agents
func (m *InMemoryMessenger) Broadcast(ctx context.Context, content string, messageType string, metadata map[string]interface{}) ([]string, error) {
	msg := NewMessage(m.agentID, "", "", content, messageType, metadata)
	globalRegistry.AddMessage(*msg)
	return []string{msg.ID}, nil
}

// Subscribe subscribes to a topic
func (m *InMemoryMessenger) Subscribe(ctx context.Context, topicName string) error {
	topic, err := globalRegistry.GetTopic(topicName)
	if err != nil {
		return err
	}

	if !topic.AddSubscriber(m.agentID) {
		return errors.New("already subscribed to this topic")
	}

	return nil
}

// Unsubscribe unsubscribes from a topic
func (m *InMemoryMessenger) Unsubscribe(ctx context.Context, topicName string) error {
	topic, err := globalRegistry.GetTopic(topicName)
	if err != nil {
		return err
	}

	if !topic.RemoveSubscriber(m.agentID) {
		return errors.New("not subscribed to this topic")
	}

	return nil
}

// CreateTopic creates a new topic
func (m *InMemoryMessenger) CreateTopic(ctx context.Context, name string, description string) error {
	topic := NewTopic(name, description, m.agentID)
	return globalRegistry.AddTopic(topic)
}

// GetMessages retrieves messages for the agent
func (m *InMemoryMessenger) GetMessages(ctx context.Context, since time.Time) ([]Message, error) {
	messages := globalRegistry.GetMessagesForAgent(m.agentID, since)
	return messages, nil
}

// GetTopics retrieves all available topics
func (m *InMemoryMessenger) GetTopics(ctx context.Context) ([]Topic, error) {
	allTopics := globalRegistry.GetAllTopics()
	topics := make([]Topic, len(allTopics))

	for i, t := range allTopics {
		topics[i] = *t
	}

	return topics, nil
}

// GetAgentID returns the agent ID associated with this messenger
func (m *InMemoryMessenger) GetAgentID() string {
	return m.agentID
}

// generateMessageID creates a unique message ID
func generateMessageID() string {
	return time.Now().Format("20060102150405") + "-" + randomString(8)
}

// randomString generates a random string of the given length
func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[rand.Intn(len(charset))]
	}
	return string(b)
}
