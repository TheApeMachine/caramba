package message

type Role string

const (
	SystemRole    Role = "system"
	UserRole      Role = "user"
	AssistantRole Role = "assistant"
	ToolRole      Role = "tool"
)

func (r Role) String() string {
	return string(r)
}
