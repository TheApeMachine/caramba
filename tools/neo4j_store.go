package tools

import (
	"context"
	"encoding/json"
	"io"

	"github.com/invopop/jsonschema"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/theapemachine/errnie"
)

type Neo4jStore struct {
	ctx       context.Context         `json:"-"`
	client    neo4j.DriverWithContext `json:"-"`
	Cypher    string                  `json:"cypher" jsonschema:"title=Cypher Query,description=The Cypher query to create or update relationships,required"`
	Reasoning string                  `json:"reasoning" jsonschema:"title=Reasoning,description=Explanation of why these relationships should be stored"`
}

func NewNeo4jStore() *Neo4jStore {
	neo4jTool := NewNeo4j()
	return &Neo4jStore{
		ctx:    context.Background(),
		client: neo4jTool.client,
	}
}

func (n *Neo4jStore) GenerateSchema() string {
	schema := jsonschema.Reflect(n)
	return string(errnie.SafeMust(func() ([]byte, error) {
		return json.MarshalIndent(schema, "", "  ")
	}))
}

func (n *Neo4jStore) Initialize() error {
	return nil
}

func (n *Neo4jStore) Connect(rw io.ReadWriteCloser) error {
	return nil
}

func (n *Neo4jStore) Use(args map[string]any) string {
	var (
		cypher  string
		ok      bool
		err error
		results neo4j.ResultWithContext
	)

	if cypher, ok = args["cypher"].(string); !ok {
		return "cypher is required"
	}

	session := n.client.NewSession(n.ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close(n.ctx)

	if results, err = session.Run(n.ctx, cypher, nil); err != nil {
		return err.Error()
	}

	if err := results.Err(); err != nil {
		return err.Error()
	}

	return "relationships stored in graph database"
}
