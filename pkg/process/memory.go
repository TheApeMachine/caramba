package process

import (
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/agent/util"
)

type MemoryLookup struct {
	Questions []string `json:"questions" jsonschema:"description=The questions to ask the memory,required"`
	Keywords  []string `json:"keywords" jsonschema:"description=The keywords to search for in the memory,required"`
	Cypher    string   `json:"cypher" jsonschema:"description=The Cypher query to execute,required"`
}

func (process *MemoryLookup) Name() string {
	return "memory_lookup"
}

func (process *MemoryLookup) Description() string {
	return "Lookup information from the memory"
}

func (process *MemoryLookup) Schema() any {
	return util.GenerateSchema[MemoryLookup]()
}

func (process *MemoryLookup) String() string {
	return fmt.Sprintf(
		"Questions:\n%s\nKeywords:\n%s\nCypher:\n%s",
		strings.Join(process.Questions, "\n"),
		strings.Join(process.Keywords, "\n"),
		process.Cypher,
	)
}

type MemoryMutate struct {
	Documents     []Document     `json:"documents" jsonschema:"description=The documents to store in vector memory,required"`
	Relationships []Relationship `json:"relationships" jsonschema:"description=The relationships to store in graph memory,required"`
}

type Document struct {
	Text     string            `json:"text" jsonschema:"description=The text of the document,required"`
	Metadata map[string]string `json:"metadata" jsonschema:"description=The metadata of the document,required"`
}

type Relationship struct {
	Source Node   `json:"source" jsonschema:"description=The source node of the relationship,required"`
	Target Node   `json:"target" jsonschema:"description=The target node of the relationship,required"`
	Type   string `json:"type" jsonschema:"description=The type of the relationship,required"`
}

type Node struct {
	ID       string            `json:"id" jsonschema:"description=The ID of the node,required"`
	Text     string            `json:"text" jsonschema:"description=The text of the node,required"`
	Metadata map[string]string `json:"metadata" jsonschema:"description=The metadata of the node,required"`
}

func (process *MemoryMutate) Name() string {
	return "memory_mutate"
}

func (process *MemoryMutate) Description() string {
	return "Store information in the memory"
}

func (process *MemoryMutate) Schema() any {
	return util.GenerateSchema[MemoryMutate]()
}

func (process *MemoryMutate) String() string {
	var out strings.Builder

	out.WriteString("Documents:\n")

	for _, document := range process.Documents {
		out.WriteString(fmt.Sprintf("%s\n", document.Text))
	}

	out.WriteString("Relationships:\n")

	for _, relationship := range process.Relationships {
		out.WriteString(fmt.Sprintf(
			"%s -[%s]-> %s\n",
			relationship.Source.Text,
			relationship.Type,
			relationship.Target.Text,
		))
	}

	return out.String()
}
