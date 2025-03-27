package memory

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	sdk "github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Neo4j struct {
	client sdk.DriverWithContext
	buffer *stream.Buffer
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

	neo4j := &Neo4j{
		client: driver,
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("memory.Neo4j.buffer")

			cypher := datura.GetMetaValue[string](artifact, "cypher")
			keywords := strings.Split(datura.GetMetaValue[string](artifact, "keywords"), ",")

			// If we have a direct cypher query, execute it
			if cypher != "" {
				// Check if this is a write operation (CREATE, MERGE, SET, DELETE, etc.)
				isWriteOperation := strings.Contains(strings.ToUpper(cypher), "CREATE") ||
					strings.Contains(strings.ToUpper(cypher), "MERGE") ||
					strings.Contains(strings.ToUpper(cypher), "SET") ||
					strings.Contains(strings.ToUpper(cypher), "DELETE")

				result, err := sdk.ExecuteQuery(
					context.Background(),
					driver,
					cypher,
					map[string]any{},
					sdk.EagerResultTransformer,
					sdk.ExecuteQueryWithDatabase("neo4j"),
				)
				if err != nil {
					return errnie.Error(err)
				}

				// For write operations, we don't need to format the results
				if isWriteOperation {
					return nil
				}

				artifact.SetMetaValue("output", formatRelationships(result.Records))
			}

			// If we have keywords, search for nodes and their relationships
			if len(keywords) > 0 && keywords[0] != "" {
				query := `
					MATCH (n)-[r]->(m)
					WHERE any(keyword IN $keywords WHERE 
						any(prop in keys(n) WHERE toString(n[prop]) CONTAINS keyword) OR
						any(prop in keys(m) WHERE toString(m[prop]) CONTAINS keyword))
					RETURN n, r, m
				`
				result, err := sdk.ExecuteQuery(
					context.Background(),
					driver,
					query,
					map[string]any{"keywords": keywords},
					sdk.EagerResultTransformer,
					sdk.ExecuteQueryWithDatabase("neo4j"),
				)
				if err != nil {
					return errnie.Error(err)
				}

				// If there's existing output, combine it with the new results
				existingOutput := datura.GetMetaValue[string](artifact, "output")
				newOutput := formatRelationships(result.Records)

				if existingOutput != "" {
					// Remove the closing tag from existing output and opening tag from new output
					existingOutput = strings.TrimSuffix(existingOutput, "</relationships>")
					newOutput = strings.TrimPrefix(newOutput, "<relationships>")
					artifact.SetMetaValue("output", existingOutput+"\n"+newOutput)
				} else {
					artifact.SetMetaValue("output", newOutput)
				}
			}

			return nil
		}),
	}

	return neo4j
}

func (n4j *Neo4j) Read(p []byte) (n int, err error) {
	errnie.Debug("Neo4j.Read")
	return n4j.buffer.Read(p)
}

func (n4j *Neo4j) Write(p []byte) (n int, err error) {
	errnie.Debug("Neo4j.Write")
	return n4j.buffer.Write(p)
}

func (n4j *Neo4j) Close() error {
	errnie.Debug("Neo4j.Close")

	if n4j.client != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		if err := n4j.client.Close(ctx); err != nil {
			return errnie.Error(err)
		}
	}

	if n4j.buffer != nil {
		return n4j.buffer.Close()
	}

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
