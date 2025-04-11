package memory

import (
	"context"
	"fmt"
	"os"
	"strings"

	sdk "github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Neo4j struct {
	ctx    context.Context
	cancel context.CancelFunc
	client sdk.DriverWithContext
}

func NewNeo4j() *Neo4j {
	errnie.Debug("memory.NewNeo4j")

	driver, err := sdk.NewDriverWithContext(
		os.Getenv("NEO4J_URL"),
		sdk.BasicAuth(
			os.Getenv("NEO4J_USERNAME"),
			os.Getenv("NEO4J_PASSWORD"),
			"",
		),
	)

	if err != nil {
		errnie.Error(err)
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	neo4j := &Neo4j{
		ctx:    ctx,
		cancel: cancel,
		client: driver,
	}

	return neo4j
}

func (n4j *Neo4j) ID() string {
	return "neo4j"
}

func (n4j *Neo4j) Get(key string) (value string, err error) {
	return "", nil
}

func (n4j *Neo4j) Put(key string, value string) (err error) {
	return nil
}

// formatRelationships takes a list of records and formats them into a relationship string
func formatRelationships(records []*sdk.Record) string {
	var relationships []string

	for _, record := range records {
		// Try to extract path or relationship pattern from the record
		for _, value := range record.Values {
			switch v := value.(type) {
			case sdk.Path:
				// Handle full paths
				var path []string
				for i, node := range v.Nodes {
					if i > 0 {
						rel := v.Relationships[i-1]
						path = append(path, fmt.Sprintf("-[%s]->", rel.Type))
					}
					path = append(path, fmt.Sprintf("%v", node.Props["name"]))
				}
				relationships = append(relationships, strings.Join(path, " "))
			case sdk.Relationship:
				// Handle single relationships
				relationships = append(relationships, fmt.Sprintf("%v -[%s]-> %v",
					v.StartElementId, v.Type, v.EndElementId))
			case sdk.Node:
				// Handle single nodes (just in case)
				relationships = append(relationships, fmt.Sprintf("%v", v.Props["name"]))
			}
		}
	}

	if len(relationships) == 0 {
		return "<relationships>\nNo relationships found\n</relationships>"
	}

	return fmt.Sprintf("<relationships>\n%s\n</relationships>",
		strings.Join(relationships, "\n"))
}
