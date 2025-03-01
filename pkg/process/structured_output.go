package process

type StructuredOutput interface {
	Name() string
	Description() string
	Schema() any
	String() string
}
