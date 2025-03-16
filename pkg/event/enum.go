package event

type Type string
type Role string

const (
	ErrorEvent    Type = "error"
	MessageEvent  Type = "message"
	ContextEvent  Type = "context"
	ToolCallEvent Type = "tool_call"
)

const (
	SystemRole    Role = "system"
	UserRole      Role = "user"
	AssistantRole Role = "assistant"
	ToolRole      Role = "tool"
)

func (t Type) String() string {
	return string(t)
}

func (r Role) String() string {
	return string(r)
}
