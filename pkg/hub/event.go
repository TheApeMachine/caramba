package hub

type TopicType string
type EventType string

const (
	TopicTypeAgent   TopicType = "agent"
	TopicTypeMessage TopicType = "message"
	TopicTypeStore   TopicType = "store"
	TopicTypeLog     TopicType = "log"
	TopicTypeTask    TopicType = "tasks"

	EventTypeQuery     EventType = "query"
	EventTypeQuestion  EventType = "question"
	EventTypeKeyword   EventType = "keyword"
	EventTypeCypher    EventType = "cypher"
	EventTypeRelation  EventType = "relation"
	EventTypeMutation  EventType = "mutation"
	EventTypeStatus    EventType = "status"
	EventTypeResponse  EventType = "response"
	EventTypeChunk     EventType = "chunk"
	EventTypePrompt    EventType = "prompt"
	EventTypeToolCall  EventType = "toolcall"
	EventTypeError     EventType = "error"
	EventTypeSuccess   EventType = "success"
	EventTypeWarning   EventType = "warning"
	EventTypeInfo      EventType = "info"
	EventTypeSystem    EventType = "system"
	EventTypeUser      EventType = "user"
	EventTypeAssistant EventType = "assistant"
)

type Event struct {
	Topic   TopicType
	Type    EventType
	Origin  string
	Message string
	Meta    map[string]string
}
