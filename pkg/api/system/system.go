package system

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/api/provider"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

// CapnpSystemTool implements the Cap'n Proto SystemTool interface
type CapnpSystemTool struct {
	agents    map[string]*AgentInfo
	topics    map[string]*TopicInfo
	queue     *MessageQueue
	mu        sync.RWMutex
	onMessage func(topic string, msg string)
}

type AgentInfo struct {
	id       string
	name     string
	status   Agent_Status
	topics   []string
	stopChan chan struct{}
}

type TopicInfo struct {
	name        string
	subscribers []string
	msgCount    uint64
}

// MessageQueue handles reliable message delivery between agents and topics
type MessageQueue struct {
	buffer    *stream.Buffer
	messages  chan *Message
	done      chan struct{}
	batchSize int
}

type Message struct {
	topic     string
	content   string
	metadata  []MetadataEntry
	timestamp int64
}

type MetadataEntry struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// NewSystemTool creates a new Cap'n Proto system tool
func NewSystemTool() (SystemTool, error) {
	return SystemTool_ServerToClient(&CapnpSystemTool{
		agents: make(map[string]*AgentInfo),
		topics: make(map[string]*TopicInfo),
		queue:  NewMessageQueue(1024), // Buffer size of 1024 messages
	}), nil
}

// Process implements the SystemTool.process method
func (s *CapnpSystemTool) Process(ctx context.Context, call SystemTool_process) error {
	cmd, err := call.Args().Command()
	if err != nil {
		return errnie.Error(err)
	}

	// Allocate results
	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	// Create result struct
	result, err := NewResult(results.Segment())
	if err != nil {
		return errnie.Error(err)
	}

	// Process based on command type
	switch cmd.Type() {
	case Command_Type_listAgents:
		if err := s.handleListAgents(ctx, &result); err != nil {
			return errnie.Error(err)
		}
	case Command_Type_listTopics:
		if err := s.handleListTopics(ctx, &result); err != nil {
			return errnie.Error(err)
		}
	case Command_Type_sendSignal:
		if err := s.handleSendSignal(ctx, cmd, &result); err != nil {
			return errnie.Error(err)
		}
	case Command_Type_breakLoop:
		if err := s.handleBreakLoop(ctx, cmd, &result); err != nil {
			return errnie.Error(err)
		}
	default:
		return errnie.Error(fmt.Errorf("unknown command type: %v", cmd.Type()))
	}

	return results.SetResult(result)
}

// ListAgents implements the SystemTool.listAgents method
func (s *CapnpSystemTool) ListAgents(ctx context.Context, call SystemTool_listAgents) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	// Create agent list
	agents, err := NewAgent_List(results.Segment(), int32(len(s.agents)))
	if err != nil {
		return errnie.Error(err)
	}

	// Fill agent list
	i := 0
	for _, info := range s.agents {
		agent := agents.At(i)
		if err := agent.SetId(info.id); err != nil {
			return errnie.Error(err)
		}
		if err := agent.SetName(info.name); err != nil {
			return errnie.Error(err)
		}
		agent.SetStatus(info.status)

		// Set topics
		topics, err := capnp.NewTextList(results.Segment(), int32(len(info.topics)))
		if err != nil {
			return errnie.Error(err)
		}
		for j, topic := range info.topics {
			topics.Set(j, topic)
		}
		if err := agent.SetTopics(topics); err != nil {
			return errnie.Error(err)
		}
		i++
	}

	return results.SetAgents(agents)
}

// ListTopics implements the SystemTool.listTopics method
func (s *CapnpSystemTool) ListTopics(ctx context.Context, call SystemTool_listTopics) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	// Create topic list
	topics, err := NewTopic_List(results.Segment(), int32(len(s.topics)))
	if err != nil {
		return errnie.Error(err)
	}

	// Fill topic list
	i := 0
	for _, info := range s.topics {
		topic := topics.At(i)
		if err := topic.SetName(info.name); err != nil {
			return errnie.Error(err)
		}

		// Set subscribers
		subs, err := capnp.NewTextList(results.Segment(), int32(len(info.subscribers)))
		if err != nil {
			return errnie.Error(err)
		}
		for j, sub := range info.subscribers {
			subs.Set(j, sub)
		}
		if err := topic.SetSubscribers(subs); err != nil {
			return errnie.Error(err)
		}

		topic.SetMessageCount(info.msgCount)
		i++
	}

	return results.SetTopics(topics)
}

// SendSignal implements the SystemTool.sendSignal method
func (s *CapnpSystemTool) SendSignal(ctx context.Context, call SystemTool_sendSignal) error {
	signal, err := call.Args().Signal()
	if err != nil {
		return errnie.Error(err)
	}

	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	result, err := NewResult(results.Segment())
	if err != nil {
		return errnie.Error(err)
	}

	targetId, err := signal.TargetId()
	if err != nil {
		return errnie.Error(err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	switch signal.Type() {
	case Signal_Type_message:
		// Handle topic message using queue
		if topic, exists := s.topics[targetId]; exists {
			payload, err := signal.Payload()
			if err != nil {
				return errnie.Error(err)
			}

			// Create message artifact
			msgArtifact := datura.New(
				datura.WithRole(datura.ArtifactRoleSignal),
				datura.WithScope(datura.ArtifactScopeMessage),
			)
			msgArtifact.SetMetaValue("topic", targetId)
			msgArtifact.SetMetaValue("content", payload)

			// Add metadata if present
			if metadata, err := signal.Metadata(); err == nil {
				var entries []MetadataEntry
				for i := 0; i < metadata.Len(); i++ {
					md := metadata.At(i)
					if key, err := md.Key(); err == nil {
						if value, err := md.Value(); err == nil {
							entries = append(entries, MetadataEntry{Key: key, Value: value})
						}
					}
				}
				metadataJSON, err := json.Marshal(entries)
				if err == nil {
					msgArtifact.SetMetaValue("metadata", string(metadataJSON))
				}
			}

			// Write to queue
			if data, err := msgArtifact.Message().Marshal(); err != nil {
				result.SetSuccess(false)
				result.SetError(fmt.Sprintf("failed to marshal message: %v", err))
				return nil
			} else if _, err := s.queue.buffer.Write(data); err != nil {
				result.SetSuccess(false)
				result.SetError(fmt.Sprintf("failed to queue message: %v", err))
				return nil
			}

			topic.msgCount++
			result.SetSuccess(true)
		} else {
			result.SetSuccess(false)
			result.SetError(fmt.Sprintf("topic not found: %s", targetId))
		}

	case Signal_Type_stop, Signal_Type_pause, Signal_Type_resume:
		// Handle agent control signals
		if agent, exists := s.agents[targetId]; exists {
			switch signal.Type() {
			case Signal_Type_stop:
				close(agent.stopChan)
				agent.status = Agent_Status_stopped
			case Signal_Type_pause:
				agent.status = Agent_Status_stopped
			case Signal_Type_resume:
				agent.status = Agent_Status_running
			}
			result.SetSuccess(true)
		} else {
			result.SetSuccess(false)
			result.SetError(fmt.Sprintf("agent not found: %s", targetId))
		}

	default:
		result.SetSuccess(false)
		result.SetError(fmt.Sprintf("unsupported signal type: %v", signal.Type()))
	}

	return results.SetResult(result)
}

// BreakLoop implements the SystemTool.breakLoop method
func (s *CapnpSystemTool) BreakLoop(ctx context.Context, call SystemTool_breakLoop) error {
	agentId, err := call.Args().AgentId()
	if err != nil {
		return errnie.Error(err)
	}

	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	result, err := NewResult(results.Segment())
	if err != nil {
		return errnie.Error(err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if agent, exists := s.agents[agentId]; exists {
		close(agent.stopChan)
		delete(s.agents, agentId)
		result.SetSuccess(true)
	} else {
		result.SetSuccess(false)
		result.SetError(fmt.Sprintf("agent not found: %s", agentId))
	}

	return results.SetResult(result)
}

// GetSchema implements the SystemTool.getSchema method
func (s *CapnpSystemTool) GetSchema(ctx context.Context, call SystemTool_getSchema) error {
	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	// Create tool schema
	tool, err := provider.NewTool(results.Segment())
	if err != nil {
		return errnie.Error(err)
	}

	// Set function details
	fn, err := tool.NewFunction()
	if err != nil {
		return errnie.Error(err)
	}

	if err := fn.SetName("system"); err != nil {
		return errnie.Error(err)
	}
	if err := fn.SetDescription("Manage system operations, agents, and topics"); err != nil {
		return errnie.Error(err)
	}

	return tool.SetFunction(fn)
}

// Helper methods

func (s *CapnpSystemTool) handleListAgents(ctx context.Context, result *Result) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Create agent list
	agents, err := NewAgent_List(result.Segment(), int32(len(s.agents)))
	if err != nil {
		return errnie.Error(err)
	}

	// Fill agent list (same as ListAgents implementation)
	i := 0
	for _, info := range s.agents {
		agent := agents.At(i)
		if err := agent.SetId(info.id); err != nil {
			return errnie.Error(err)
		}
		if err := agent.SetName(info.name); err != nil {
			return errnie.Error(err)
		}
		agent.SetStatus(info.status)

		topics, err := capnp.NewTextList(result.Segment(), int32(len(info.topics)))
		if err != nil {
			return errnie.Error(err)
		}
		for j, topic := range info.topics {
			topics.Set(j, topic)
		}
		if err := agent.SetTopics(topics); err != nil {
			return errnie.Error(err)
		}
		i++
	}

	result.SetSuccess(true)
	return result.SetData(agents.ToPtr())
}

func (s *CapnpSystemTool) handleListTopics(ctx context.Context, result *Result) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Create topic list
	topics, err := NewTopic_List(result.Segment(), int32(len(s.topics)))
	if err != nil {
		return errnie.Error(err)
	}

	// Fill topic list (same as ListTopics implementation)
	i := 0
	for _, info := range s.topics {
		topic := topics.At(i)
		if err := topic.SetName(info.name); err != nil {
			return errnie.Error(err)
		}

		subs, err := capnp.NewTextList(result.Segment(), int32(len(info.subscribers)))
		if err != nil {
			return errnie.Error(err)
		}
		for j, sub := range info.subscribers {
			subs.Set(j, sub)
		}
		if err := topic.SetSubscribers(subs); err != nil {
			return errnie.Error(err)
		}

		topic.SetMessageCount(info.msgCount)
		i++
	}

	result.SetSuccess(true)
	return result.SetData(topics.ToPtr())
}

func (s *CapnpSystemTool) handleSendSignal(ctx context.Context, cmd Command, result *Result) error {
	payload, err := cmd.Payload()
	if err != nil {
		return errnie.Error(err)
	}

	signal := Signal(payload.Struct())
	if !signal.IsValid() {
		return errnie.Error(fmt.Errorf("invalid signal payload"))
	}

	targetId, err := signal.TargetId()
	if err != nil {
		return errnie.Error(err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	switch signal.Type() {
	case Signal_Type_message:
		// Handle topic message using queue
		if topic, exists := s.topics[targetId]; exists {
			payload, err := signal.Payload()
			if err != nil {
				return errnie.Error(err)
			}

			// Create message artifact
			msgArtifact := datura.New(
				datura.WithRole(datura.ArtifactRoleSignal),
				datura.WithScope(datura.ArtifactScopeMessage),
			)
			msgArtifact.SetMetaValue("topic", targetId)
			msgArtifact.SetMetaValue("content", payload)

			// Add metadata if present
			if metadata, err := signal.Metadata(); err == nil {
				var entries []MetadataEntry
				for i := 0; i < metadata.Len(); i++ {
					md := metadata.At(i)
					if key, err := md.Key(); err == nil {
						if value, err := md.Value(); err == nil {
							entries = append(entries, MetadataEntry{Key: key, Value: value})
						}
					}
				}
				metadataJSON, err := json.Marshal(entries)
				if err == nil {
					msgArtifact.SetMetaValue("metadata", string(metadataJSON))
				}
			}

			// Write to queue
			if data, err := msgArtifact.Message().Marshal(); err != nil {
				result.SetSuccess(false)
				result.SetError(fmt.Sprintf("failed to marshal message: %v", err))
				return nil
			} else if _, err := s.queue.buffer.Write(data); err != nil {
				result.SetSuccess(false)
				result.SetError(fmt.Sprintf("failed to queue message: %v", err))
				return nil
			}

			topic.msgCount++
			result.SetSuccess(true)
		} else {
			result.SetSuccess(false)
			result.SetError(fmt.Sprintf("topic not found: %s", targetId))
		}

	case Signal_Type_stop, Signal_Type_pause, Signal_Type_resume:
		// Handle agent control signals
		if agent, exists := s.agents[targetId]; exists {
			switch signal.Type() {
			case Signal_Type_stop:
				close(agent.stopChan)
				agent.status = Agent_Status_stopped
			case Signal_Type_pause:
				agent.status = Agent_Status_stopped
			case Signal_Type_resume:
				agent.status = Agent_Status_running
			}
			result.SetSuccess(true)
		} else {
			result.SetSuccess(false)
			result.SetError(fmt.Sprintf("agent not found: %s", targetId))
		}

	default:
		result.SetSuccess(false)
		result.SetError(fmt.Sprintf("unsupported signal type: %v", signal.Type()))
	}

	return nil
}

func (s *CapnpSystemTool) handleBreakLoop(ctx context.Context, cmd Command, result *Result) error {
	payload, err := cmd.Payload()
	if err != nil {
		return errnie.Error(err)
	}

	agentId := payload.Text()

	s.mu.Lock()
	defer s.mu.Unlock()

	if agent, exists := s.agents[agentId]; exists {
		close(agent.stopChan)
		delete(s.agents, agentId)
		result.SetSuccess(true)
	} else {
		result.SetSuccess(false)
		result.SetError(fmt.Sprintf("agent not found: %s", agentId))
	}

	return nil
}

// Public API for managing agents and topics

func (s *CapnpSystemTool) RegisterAgent(id, name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.agents[id]; exists {
		return errnie.Error(fmt.Errorf("agent already exists: %s", id))
	}

	s.agents[id] = &AgentInfo{
		id:       id,
		name:     name,
		status:   Agent_Status_running,
		topics:   make([]string, 0),
		stopChan: make(chan struct{}),
	}

	return nil
}

func (s *CapnpSystemTool) RegisterTopic(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.topics[name]; exists {
		return errnie.Error(fmt.Errorf("topic already exists: %s", name))
	}

	s.topics[name] = &TopicInfo{
		name:        name,
		subscribers: make([]string, 0),
		msgCount:    0,
	}

	return nil
}

func (s *CapnpSystemTool) Subscribe(agentId, topic string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	agent, exists := s.agents[agentId]
	if !exists {
		return errnie.Error(fmt.Errorf("agent not found: %s", agentId))
	}

	topicInfo, exists := s.topics[topic]
	if !exists {
		return errnie.Error(fmt.Errorf("topic not found: %s", topic))
	}

	// Add topic to agent's subscriptions
	agent.topics = append(agent.topics, topic)

	// Add agent to topic's subscribers
	topicInfo.subscribers = append(topicInfo.subscribers, agentId)

	return nil
}

func (s *CapnpSystemTool) SetMessageHandler(handler func(topic string, msg string)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onMessage = handler
}

// NewMessageQueue creates a new message queue with the specified buffer size
func NewMessageQueue(bufferSize int) *MessageQueue {
	mq := &MessageQueue{
		messages:  make(chan *Message, bufferSize),
		done:      make(chan struct{}),
		batchSize: 100, // Process messages in batches
	}

	mq.buffer = stream.NewBuffer(func(artifact *datura.Artifact) error {
		msg := &Message{
			topic:     datura.GetMetaValue[string](artifact, "topic"),
			content:   datura.GetMetaValue[string](artifact, "content"),
			timestamp: time.Now().UnixNano(),
		}

		// Add metadata if present
		if metadataStr := datura.GetMetaValue[string](artifact, "metadata"); metadataStr != "" {
			var metadata []MetadataEntry
			if err := json.Unmarshal([]byte(metadataStr), &metadata); err != nil {
				return errnie.Error(err)
			}
			msg.metadata = metadata
		}

		select {
		case mq.messages <- msg:
			return nil
		default:
			return errnie.Error(fmt.Errorf("message queue full"))
		}
	})

	// Start message processing goroutine
	go mq.processMessages()

	return mq
}

func (mq *MessageQueue) processMessages() {
	batch := make([]*Message, 0, mq.batchSize)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-mq.done:
			return
		case msg := <-mq.messages:
			batch = append(batch, msg)
			if len(batch) >= mq.batchSize {
				mq.deliverBatch(batch)
				batch = batch[:0]
			}
		case <-ticker.C:
			if len(batch) > 0 {
				mq.deliverBatch(batch)
				batch = batch[:0]
			}
		}
	}
}

func (mq *MessageQueue) deliverBatch(batch []*Message) {
	// Group messages by topic for efficient delivery
	topicMessages := make(map[string][]*Message)
	for _, msg := range batch {
		topicMessages[msg.topic] = append(topicMessages[msg.topic], msg)
	}

	// Deliver messages to each topic
	for topic, messages := range topicMessages {
		if err := mq.deliverToTopic(topic, messages); err != nil {
			errnie.Error(fmt.Errorf("failed to deliver messages to topic %s: %v", topic, err))
		}
	}
}

func (mq *MessageQueue) deliverToTopic(topic string, messages []*Message) error {
	// Create a new artifact for the batch
	artifact := datura.New(
		datura.WithRole(datura.ArtifactRoleSignal),
		datura.WithScope(datura.ArtifactScopeMessage),
	)

	// Set topic and messages
	artifact.SetMetaValue("topic", topic)
	messagesJSON, err := json.Marshal(messages)
	if err != nil {
		return errnie.Error(err)
	}
	artifact.SetMetaValue("messages", string(messagesJSON))

	// Write to buffer for processing
	data, err := artifact.Message().Marshal()
	if err != nil {
		return errnie.Error(err)
	}
	_, err = mq.buffer.Write(data)
	return errnie.Error(err)
}

func (mq *MessageQueue) Close() error {
	close(mq.done)
	return mq.buffer.Close()
}
