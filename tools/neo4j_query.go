package tools

import (
	"context"
	"encoding/json"
	"io"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/theapemachine/caramba/utils"
)

type Neo4jQuery struct {
	ctx       context.Context         `json:"-"`
	client    neo4j.DriverWithContext `json:"-"`
	Cypher    string                  `json:"cypher" jsonschema:"title=Cypher Query,description=The Cypher query to execute,required"`
	Reasoning string                  `json:"reasoning" jsonschema:"title=Reasoning,description=Explanation of why this query pattern is relevant"`
}

func NewNeo4jQuery() *Neo4jQuery {
	return &Neo4jQuery{
		ctx: context.Background(),
	}
}

func (n *Neo4jQuery) Name() string {
	return "neo4j"
}

func (n *Neo4jQuery) Description() string {
	return "Interact with Neo4j"
}

func (n *Neo4jQuery) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Neo4jQuery]()
}

func (n *Neo4jQuery) Initialize() error {
	neo4jTool := NewNeo4j()
	n.client = neo4jTool.client
	return nil
}

func (n *Neo4jQuery) Connect(ctx context.Context, rw io.ReadWriteCloser) error {
	return nil
}

func (n *Neo4jQuery) Use(ctx context.Context, args map[string]any) string {
	var (
		cypher  string
		ok      bool
		results neo4j.ResultWithContext
		err     error
		buf     []byte
	)

	if cypher, ok = args["cypher"].(string); !ok {
		return "cypher is required"
	}

	session := n.client.NewSession(n.ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead})
	defer session.Close(n.ctx)

	if results, err = session.Run(n.ctx, cypher, nil); err != nil {
		return err.Error()
	}

	if err := results.Err(); err != nil {
		return err.Error()
	}

	var records []map[string]interface{}
	for results.Next(n.ctx) {
		record := results.Record()
		records = append(records, record.Values[0].(neo4j.Node).Props)
	}

	if len(records) == 0 {
		return "No results found"
	}

	if buf, err = json.Marshal(records); err != nil {
		return err.Error()
	}

	return string(buf)
}
