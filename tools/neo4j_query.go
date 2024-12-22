package tools

import (
	"context"
	"encoding/json"
	"io"

	"github.com/invopop/jsonschema"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/theapemachine/errnie"
)

type Neo4jQuery struct {
	ctx       context.Context         `json:"-"`
	client    neo4j.DriverWithContext `json:"-"`
	Cypher    string                  `json:"cypher" jsonschema:"title=Cypher Query,description=The Cypher query to execute,required"`
	Reasoning string                  `json:"reasoning" jsonschema:"title=Reasoning,description=Explanation of why this query pattern is relevant"`
}

func NewNeo4jQuery() *Neo4jQuery {
	neo4jTool := NewNeo4j()
	return &Neo4jQuery{
		ctx:    context.Background(),
		client: neo4jTool.client,
	}
}

func (n *Neo4jQuery) GenerateSchema() string {
	schema := jsonschema.Reflect(n)
	return string(errnie.SafeMust(func() ([]byte, error) {
		return json.MarshalIndent(schema, "", "  ")
	}))
}

func (n *Neo4jQuery) Initialize() error {
	return nil
}

func (n *Neo4jQuery) Connect(rw io.ReadWriteCloser) error {
	return nil
}

func (n *Neo4jQuery) Use(args map[string]any) string {
	var (
		cypher string
		ok     bool
	)

	if cypher, ok = args["cypher"].(string); !ok {
		return "cypher is required"
	}

	session := n.client.NewSession(n.ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead})
	defer session.Close(n.ctx)

	result := errnie.SafeMust(func() (neo4j.ResultWithContext, error) {
		return session.Run(n.ctx, cypher, nil)
	})

	if result.Err() != nil {
		return result.Err().Error()
	}

	var records []map[string]interface{}
	for result.Next(n.ctx) {
		record := result.Record()
		records = append(records, record.Values[0].(neo4j.Node).Props)
	}

	if result == nil {
		return "something went wrong"
	}

	return string(errnie.SafeMust(func() ([]byte, error) {
		return json.Marshal(records)
	}))
}
