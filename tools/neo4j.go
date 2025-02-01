package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/invopop/jsonschema"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/theapemachine/errnie"
)

/*
Neo4j is a wrapper around the Neo4j database that turns it into a tool,
usable by the agent.
*/
type Neo4j struct {
	client    neo4j.DriverWithContext `json:"-"`
	Operation string                  `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=recall,enum=store,required"`
	Cypher    string                  `json:"cypher" jsonschema:"title=Cypher,description=The Cypher query to execute,required"`
	Reasoning string                  `json:"reasoning" jsonschema:"title=Reasoning,description=Explanation of why this relationship pattern is relevant"`
}

/*
GenerateSchema implements the Tool interface and renders the schema as a jsonschema string,
which can be injected into the prompt. It is used to explain to the agent how to use the tool.
*/
func (neo4j *Neo4j) GenerateSchema() string {
	schema := jsonschema.Reflect(&Neo4j{})
	return string(errnie.SafeMust(func() ([]byte, error) {
		return json.MarshalIndent(schema, "", "  ")
	}))
}

/*
NewNeo4j creates a new Neo4j client.
*/
func NewNeo4j() *Neo4j {
	ctx := context.Background()

	client, err := neo4j.NewDriverWithContext("neo4j://localhost:7687", neo4j.BasicAuth("neo4j", "securepassword", ""))
	if err != nil {
		return &Neo4j{}
	}

	if err := client.VerifyConnectivity(ctx); err != nil {
		return &Neo4j{}
	}

	return &Neo4j{client: client}
}

/*
Initialize initializes the Neo4j client.
*/
func (n *Neo4j) Initialize() error {
	if n.client == nil {
		return nil
	}
	ctx := context.Background()
	return n.client.VerifyConnectivity(ctx)
}

func (n *Neo4j) Connect() error {
	return nil
}

/*
Query executes a Cypher query on the Neo4j database and returns the results.
*/
func (n *Neo4j) Query(query string) (out []map[string]interface{}, err error) {
	if n.client == nil {
		return nil, fmt.Errorf("Neo4j is not available")
	}

	ctx := context.Background()
	session := n.client.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead})
	defer session.Close(ctx)

	result := errnie.SafeMust(func() (neo4j.ResultWithContext, error) {
		return session.Run(ctx, query, nil)
	})

	var records []map[string]interface{}
	for result.Next(ctx) {
		record := result.Record()
		records = append(records, record.Values[0].(neo4j.Node).Props)
	}

	errnie.MustVoid(result.Err())
	return records, nil
}

/*
Write executes a Cypher query on the Neo4j database and returns the results.
*/
func (n *Neo4j) Write(query string) neo4j.ResultWithContext {
	if n.client == nil {
		return nil
	}

	ctx := context.Background()
	session := n.client.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close(ctx)

	return errnie.SafeMust(func() (neo4j.ResultWithContext, error) {
		return session.Run(ctx, query, nil)
	})
}

/*
Close closes the Neo4j client connection.
*/
func (n *Neo4j) Close() error {
	ctx := context.Background()
	return n.client.Close(ctx)
}

/*
Use implements the Tool interface and is used to execute the tool.
*/
func (neo4j *Neo4j) Use(ctx context.Context, args map[string]any) string {
	if neo4j.client == nil {
		return "Neo4j is not available"
	}

	switch neo4j.Operation {
	case "query":
		records := errnie.SafeMust(func() ([]map[string]interface{}, error) {
			return neo4j.Query(args["cypher"].(string))
		})
		result := errnie.SafeMust(func() ([]byte, error) {
			return json.Marshal(records)
		})
		return string(result)

	case "write":
		return neo4j.Write(args["query"].(string)).Err().Error()

	default:
		return "Unsupported operation"
	}
}
