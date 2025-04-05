package memory

import (
	"context"
	"fmt"
	"os"
	"strings"

	sdk "github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/theapemachine/caramba/pkg/datura"
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

/*
Generate processes a query and returns the results via the artifact channel.
It implements the Generator pattern to handle graph database operations asynchronously.
*/
func (n4j *Neo4j) Generate(
	buffer chan *datura.ArtifactBuilder,
	fn ...func(artifact *datura.ArtifactBuilder) *datura.ArtifactBuilder,
) chan *datura.ArtifactBuilder {
	errnie.Debug("memory.Neo4j.Generate")

	out := make(chan *datura.ArtifactBuilder)

	go func() {
		defer close(out)

		select {
		case <-n4j.ctx.Done():
			errnie.Debug("memory.Neo4j.Generate.ctx.Done")
			n4j.cancel()
			return
		case artifact := <-buffer:
			// Extract query information from the artifact
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
					n4j.client,
					cypher,
					map[string]any{},
					sdk.EagerResultTransformer,
					sdk.ExecuteQueryWithDatabase("neo4j"),
				)
				if err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}

				// For write operations, we don't need to format the results
				if isWriteOperation {
					out <- datura.New(datura.WithPayload([]byte("Write operation completed successfully")))
					return
				}

				// Format and return the results
				formattedResults := formatRelationships(result.Records)
				resultArtifact := datura.New(datura.WithPayload([]byte(formattedResults)))
				resultArtifact.SetMetaValue("output", formattedResults)
				out <- resultArtifact
				return
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
					n4j.client,
					query,
					map[string]any{"keywords": keywords},
					sdk.EagerResultTransformer,
					sdk.ExecuteQueryWithDatabase("neo4j"),
				)
				if err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}

				// Format the results
				newOutput := formatRelationships(result.Records)
				existingOutput := datura.GetMetaValue[string](artifact, "output")

				var finalOutput string
				if existingOutput != "" {
					// Combine existing and new results
					existingOutput = strings.TrimSuffix(existingOutput, "</relationships>")
					newOutput = strings.TrimPrefix(newOutput, "<relationships>")
					finalOutput = existingOutput + "\n" + newOutput
				} else {
					finalOutput = newOutput
				}

				resultArtifact := datura.New(datura.WithPayload([]byte(finalOutput)))
				resultArtifact.SetMetaValue("output", finalOutput)
				out <- resultArtifact
				return
			}

			// If no valid operation was specified
			out <- datura.New(datura.WithError(errnie.Error(fmt.Errorf("no valid operation specified"))))
		}
	}()

	return out
}

func (n4j *Neo4j) Name() string {
	return "neo4j"
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
