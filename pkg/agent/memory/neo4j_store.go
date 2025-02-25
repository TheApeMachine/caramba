package memory

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

/*
GraphStore interface defines operations for graph-based memory storage.
This interface abstracts the graph database operations for storing and
retrieving nodes, relationships, and executing graph queries.
*/
type GraphStore interface {
	/*
		CreateNode creates a new node in the graph.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - id: The unique identifier for the node
		  - labels: Classification labels for the node
		  - properties: Key-value properties to store with the node

		Returns:
		  - An error if the operation fails, or nil on success
	*/
	CreateNode(ctx context.Context, id string, labels []string, properties map[string]interface{}) error

	/*
		CreateRelationship creates a relationship between two nodes.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - fromID: The source node ID
		  - toID: The target node ID
		  - relType: The type of relationship to create
		  - properties: Key-value properties to store with the relationship

		Returns:
		  - An error if the operation fails, or nil on success
	*/
	CreateRelationship(ctx context.Context, fromID, toID, relType string, properties map[string]interface{}) error

	/*
		Query executes a cypher query against the graph.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - query: The Cypher query string to execute
		  - params: Parameters for the query

		Returns:
		  - Results from the query as a slice of maps
		  - An error if the operation fails, or nil on success
	*/
	Query(ctx context.Context, query string, params map[string]interface{}) ([]map[string]interface{}, error)

	/*
		DeleteNode removes a node from the graph.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - id: The unique identifier of the node to delete

		Returns:
		  - An error if the operation fails, or nil on success
	*/
	DeleteNode(ctx context.Context, id string) error
}

// Neo4jStore implements the GraphStore interface using Neo4j.
type Neo4jStore struct {
	// driver is the Neo4j driver instance
	driver neo4j.Driver
	// database is the name of the database in Neo4j
	database string
}

// NewNeo4jStore creates a new Neo4j graph store.
//
// Parameters:
//   - url: The Neo4j server URL
//   - username: The Neo4j username
//   - password: The Neo4j password
//   - database: The Neo4j database name
//
// Returns:
//   - A pointer to an initialized Neo4jStore
//   - An error if initialization fails, or nil on success
func NewNeo4jStore(url, username, password, database string) (*Neo4jStore, error) {
	if database == "" {
		database = "neo4j" // Default database name
	}

	// Create authentication config
	authToken := neo4j.BasicAuth(username, password, "")

	// Create Neo4j driver
	driver, err := neo4j.NewDriver(url, authToken)
	if err != nil {
		return nil, fmt.Errorf("failed to create Neo4j driver: %w", err)
	}

	err = driver.VerifyConnectivity()
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Neo4j: %w", err)
	}

	return &Neo4jStore{
		driver:   driver,
		database: database,
	}, nil
}

// Close closes the Neo4j driver connection
func (n *Neo4jStore) Close() error {
	return n.driver.Close()
}

// CreateNode creates a new node in the Neo4j graph.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - id: The unique identifier for the node
//   - labels: Classification labels for the node
//   - properties: Key-value properties to store with the node
//
// Returns:
//   - An error if the operation fails, or nil on success
func (n *Neo4jStore) CreateNode(ctx context.Context, id string, labels []string, properties map[string]interface{}) error {
	// Start a new session
	session := n.driver.NewSession(neo4j.SessionConfig{
		DatabaseName: n.database,
		AccessMode:   neo4j.AccessModeWrite,
	})
	defer session.Close()

	// Add the ID to properties
	props := make(map[string]interface{})
	for k, v := range properties {
		props[k] = v
	}
	props["id"] = id

	// Build label string
	labelStr := ""
	for _, label := range labels {
		labelStr += ":" + label
	}

	// Create the Cypher query
	query := fmt.Sprintf("CREATE (n%s $props)", labelStr)
	params := map[string]interface{}{
		"props": props,
	}

	// Execute the query within a transaction
	tx, err := session.BeginTransaction()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	result, err := tx.Run(query, params)
	if err != nil {
		return fmt.Errorf("failed to run query: %w", err)
	}

	_, err = result.Consume()
	if err != nil {
		return fmt.Errorf("failed to consume result: %w", err)
	}

	err = tx.Commit()
	if err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// CreateRelationship creates a relationship between two nodes in the Neo4j graph.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - fromID: The source node ID
//   - toID: The target node ID
//   - relType: The type of relationship to create
//   - properties: Key-value properties to store with the relationship
//
// Returns:
//   - An error if the operation fails, or nil on success
func (n *Neo4jStore) CreateRelationship(ctx context.Context, fromID, toID, relType string, properties map[string]interface{}) error {
	// Start a new session
	session := n.driver.NewSession(neo4j.SessionConfig{
		DatabaseName: n.database,
		AccessMode:   neo4j.AccessModeWrite,
	})
	defer session.Close()

	// Build the Cypher query
	query := `
		MATCH (a), (b)
		WHERE a.id = $fromID AND b.id = $toID
		CREATE (a)-[r:` + relType + ` $props]->(b)
		RETURN r
	`

	params := map[string]interface{}{
		"fromID": fromID,
		"toID":   toID,
		"props":  properties,
	}

	// Execute the query within a transaction
	tx, err := session.BeginTransaction()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	result, err := tx.Run(query, params)
	if err != nil {
		return fmt.Errorf("failed to run query: %w", err)
	}

	_, err = result.Consume()
	if err != nil {
		return fmt.Errorf("failed to consume result: %w", err)
	}

	err = tx.Commit()
	if err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return err
}

// Query executes a Cypher query against the Neo4j database.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - query: The Cypher query string to execute
//   - params: Parameters for the query
//
// Returns:
//   - Results from the query as a slice of maps
//   - An error if the operation fails, or nil on success
func (n *Neo4jStore) Query(ctx context.Context, query string, params map[string]interface{}) ([]map[string]interface{}, error) {
	// Start a new session
	session := n.driver.NewSession(neo4j.SessionConfig{
		DatabaseName: n.database,
		AccessMode:   neo4j.AccessModeRead,
	})
	defer session.Close()

	// Execute the query within a transaction
	result, err := session.ReadTransaction(func(tx neo4j.Transaction) (interface{}, error) {
		result, err := tx.Run(query, params)
		if err != nil {
			return nil, err
		}

		var records []map[string]interface{}
		for result.Next() {
			record := result.Record()
			recordMap := make(map[string]interface{})

			for idx, key := range record.Keys {
				recordMap[key] = record.Values[idx]
			}

			records = append(records, recordMap)
		}

		// Check for errors during result iteration
		if err := result.Err(); err != nil {
			return nil, err
		}

		return records, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}

	// Convert and return the results
	if records, ok := result.([]map[string]interface{}); ok {
		return records, nil
	}

	return nil, fmt.Errorf("unexpected result type from query")
}

// DeleteNode removes a node from the Neo4j graph.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - id: The unique identifier of the node to delete
//
// Returns:
//   - An error if the operation fails, or nil on success
func (n *Neo4jStore) DeleteNode(ctx context.Context, id string) error {
	// Start a new session
	session := n.driver.NewSession(neo4j.SessionConfig{
		DatabaseName: n.database,
		AccessMode:   neo4j.AccessModeWrite,
	})
	defer session.Close()

	// Build the Cypher query to delete the node and all its relationships
	query := `
		MATCH (n {id: $id})
		DETACH DELETE n
	`

	params := map[string]interface{}{
		"id": id,
	}

	// Execute the query within a transaction
	_, err := session.WriteTransaction(func(tx neo4j.Transaction) (interface{}, error) {
		result, err := tx.Run(query, params)
		if err != nil {
			return nil, err
		}

		// Consume the result to complete the transaction
		summary, err := result.Consume()
		if err != nil {
			return nil, err
		}

		// Return the number of nodes deleted
		return summary.Counters().NodesDeleted(), nil
	})

	if err != nil {
		return fmt.Errorf("failed to delete node: %w", err)
	}

	return nil
}

/*
RetrieveMemoriesByGraph searches for memories using graph relationships.
It executes a Cypher query against the graph store to find related memories.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - cypherQuery: The Cypher query to execute against the graph database
  - params: Parameters for the Cypher query

Returns:
  - A slice of EnhancedMemoryEntry objects matching the query
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) RetrieveMemoriesByGraph(ctx context.Context, cypherQuery string, params map[string]interface{}) ([]EnhancedMemoryEntry, error) {
	if !um.options.EnableGraphStore || um.graphStore == nil {
		return nil, errors.New("graph store not enabled or not available")
	}

	// Execute query
	results, err := um.graphStore.Query(ctx, cypherQuery, params)
	if err != nil {
		return nil, fmt.Errorf("failed to execute graph query: %w", err)
	}

	// Convert results to EnhancedMemoryEntry
	memories := make([]EnhancedMemoryEntry, 0, len(results))
	for _, result := range results {
		// Extract memory node properties
		node, ok := result["memory"].(map[string]interface{})
		if !ok {
			continue
		}

		id, _ := node["id"].(string)
		content, _ := node["content"].(string)
		agentID, _ := node["agent_id"].(string)
		source, _ := node["source"].(string)
		memType, _ := node["type"].(string)
		createdAtStr, _ := node["created_at"].(string)

		// Parse created at
		var createdAt time.Time
		if createdAtStr != "" {
			createdAt, _ = time.Parse(time.RFC3339, createdAtStr)
		}

		// Create memory entry
		entry := EnhancedMemoryEntry{
			ID:          id,
			AgentID:     agentID,
			Content:     content,
			Type:        MemoryType(memType),
			Source:      source,
			CreatedAt:   createdAt,
			AccessCount: 1,
			LastAccess:  time.Now(),
			Metadata:    node,
		}

		memories = append(memories, entry)
	}

	return memories, nil
}
