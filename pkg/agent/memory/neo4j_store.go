package memory

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

type Neo4jStore struct {
	hub      *hub.Queue
	logger   *output.Logger
	driver   neo4j.DriverWithContext
	database string
}

// NewNeo4jStore creates a new Neo4j graph store.
//
// Parameters:
//   - database: The Neo4j database name
//
// Returns:
//   - A pointer to an initialized Neo4jStore
func NewNeo4jStore(
	database string,
) *Neo4jStore {
	if database == "" {
		database = "neo4j"
	}

	logger := output.NewLogger()

	driver, err := neo4j.NewDriverWithContext(
		os.Getenv("NEO4J_URL"),
		neo4j.BasicAuth(
			os.Getenv("NEO4J_USERNAME"),
			os.Getenv("NEO4J_PASSWORD"),
			"",
		),
	)

	if err != nil {
		logger.Error("neo4j", err)
		return nil
	}

	logger.Success("neo4j", "connected")

	return &Neo4jStore{
		hub:      hub.NewQueue(),
		logger:   logger,
		driver:   driver,
		database: database,
	}
}

/*
Query constructs the Cypher query for the query and executes it.
Returns formatted relationship paths as strings.
*/
func (n *Neo4jStore) Query(ctx context.Context, query map[string]any) (string, error) {
	var results strings.Builder

	for _, term := range query["keywords"].([]string) {
		// Simple query to find relationships
		result, err := n.Cypher(
			ctx,
			`
		MATCH p=(a)-[r]->(b)
		WHERE a.name CONTAINS $term OR b.name CONTAINS $term
		RETURN a.name as source, labels(a)[0] as sourceLabel, 
		       type(r) as relationship, 
		       b.name as target, labels(b)[0] as targetLabel
		LIMIT 20
		`,
			map[string]interface{}{
				"term": term,
			},
		)

		if err != nil {
			return "", n.logger.Error(
				"neo4j",
				fmt.Errorf("failed to query: %w", err),
			)
		}

		results.WriteString(n.handleResults(result))

		if len(result) == 0 {
			return "", n.logger.Error(
				"neo4j",
				fmt.Errorf("no relationships found for: %s", term),
			)
		}
	}

	if qry, ok := query["query"].(string); ok {
		result, err := n.Cypher(ctx, qry, nil)

		if err != nil {
			return "", n.logger.Error(
				"neo4j",
				fmt.Errorf("failed to query: %w", err),
			)
		}

		results.WriteString(n.handleResults(result))
	}

	return results.String(), nil
}

func (n *Neo4jStore) handleResults(results []map[string]any) string {
	var out strings.Builder

	for _, r := range results {
		rel := fmt.Sprintf("%v:%v -[%v]-> %v:%v\n",
			r["sourceLabel"],
			r["source"],
			r["relationship"],
			r["targetLabel"],
			r["target"],
		)

		out.WriteString(rel)
		n.hub.Add(hub.NewMetric("neo4j", "relationship", rel))
	}

	return out.String()
}

/*
Mutate constructs the Cypher query for the mutation and executes it.
*/
func (n *Neo4jStore) Mutate(ctx context.Context, query map[string]any) (err error) {
	for _, entity := range query["relationships"].([]map[string]any) {
		_, err = n.Cypher(
			ctx,
			`
		CALL apoc.merge.node([$entityLabel], {name: $entityName}) YIELD node AS n
		CALL apoc.merge.node([$targetLabel], {name: $targetName}) YIELD node AS m
		CALL apoc.create.relationship(n, $relationshipType, $props, m) YIELD rel
		RETURN n, rel, m
		`,
			map[string]interface{}{
				"entityLabel":      entity["source"].(map[string]any)["text"],
				"entityName":       entity["source"].(map[string]any)["text"],
				"targetLabel":      entity["target"].(map[string]any)["text"],
				"targetName":       entity["target"].(map[string]any)["text"],
				"relationshipType": entity["type"],
				"props":            entity["source"].(map[string]any)["metadata"],
			},
		)
		if err != nil {
			n.logger.Error(
				"neo4j",
				fmt.Errorf("failed to mutate: %w", err),
			)
		}
	}

	return
}

func (n *Neo4jStore) Cypher(
	ctx context.Context,
	query string,
	params map[string]any,
) ([]map[string]any, error) {
	session := n.driver.NewSession(ctx, neo4j.SessionConfig{
		DatabaseName: n.database,
		AccessMode:   neo4j.AccessModeWrite,
	})

	defer session.Close(ctx)

	// Execute the query within a transaction
	tx, err := session.BeginTransaction(ctx)

	if err != nil {
		return nil, n.logger.Error(
			"neo4j",
			fmt.Errorf("failed to begin transaction: %w", err),
		)
	}

	result, err := tx.Run(ctx, query, params)

	if err != nil {
		return nil, n.logger.Error(
			"neo4j",
			fmt.Errorf("failed to run query: %w", err),
		)
	}

	out := make([]map[string]any, 0)

	for result.Next(ctx) {
		record := result.Record()
		out = append(out, record.AsMap())
	}

	// Still consume for the summary
	_, err = result.Consume(ctx)

	if err != nil {
		return nil, n.logger.Error(
			"neo4j",
			fmt.Errorf("failed to consume: %w", err),
		)
	}

	err = tx.Commit(ctx)

	if err != nil {
		return nil, n.logger.Error(
			"neo4j",
			fmt.Errorf("failed to commit: %w", err),
		)
	}

	return out, nil
}
