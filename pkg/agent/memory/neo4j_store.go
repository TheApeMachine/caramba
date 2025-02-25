package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
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
	// url is the connection URL for the Neo4j server
	url string
	// username is the authentication username for Neo4j
	username string
	// password is the authentication password for Neo4j
	password string
	// database is the name of the database in Neo4j
	database string
	// httpClient is the HTTP client for making API requests
	httpClient *http.Client
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

	// Ensure URL is properly formatted for Neo4j
	if !strings.HasPrefix(url, "bolt://") && !strings.HasPrefix(url, "neo4j://") {
		return nil, fmt.Errorf("invalid Neo4j URL: %s, must start with bolt:// or neo4j://", url)
	}

	store := &Neo4jStore{
		url:        url,
		username:   username,
		password:   password,
		database:   database,
		httpClient: &http.Client{},
	}

	// Test the connection
	if err := store.testConnection(); err != nil {
		return nil, fmt.Errorf("failed to connect to Neo4j: %w", err)
	}

	return store, nil
}

// testConnection checks if the Neo4j server is reachable.
func (n *Neo4jStore) testConnection() error {
	// This is a simplified check, in a real implementation you would
	// try to establish a proper Neo4j connection

	// TODO: Replace with proper Neo4j driver connection test
	// For now, we'll assume if the URL is well-formed, it's good enough
	if strings.HasPrefix(n.url, "bolt://") || strings.HasPrefix(n.url, "neo4j://") {
		return nil
	}

	return fmt.Errorf("invalid Neo4j URL format: %s", n.url)
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
	// Build the Cypher query
	labelStr := strings.Join(labels, ":")
	if labelStr != "" {
		labelStr = ":" + labelStr
	}

	// Add the ID to properties
	props := make(map[string]interface{})
	for k, v := range properties {
		props[k] = v
	}
	props["id"] = id

	// Create the query
	query := fmt.Sprintf("CREATE (n%s {props}) RETURN n", labelStr)
	params := map[string]interface{}{
		"props": props,
	}

	// Execute the query
	_, err := n.Query(ctx, query, params)
	return err
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

	// Execute the query
	_, err := n.Query(ctx, query, params)
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
	type Statement struct {
		Statement  string                 `json:"statement"`
		Parameters map[string]interface{} `json:"parameters"`
	}

	type TransactionRequest struct {
		Statements []Statement `json:"statements"`
	}

	// Build the request
	reqBody := TransactionRequest{
		Statements: []Statement{
			{
				Statement:  query,
				Parameters: params,
			},
		},
	}

	// Convert to JSON
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Build the URL
	endpoint := fmt.Sprintf("%s/db/%s/tx/commit", n.url, n.database)
	// For bolt URLs, convert to HTTP for the REST API
	endpoint = strings.Replace(endpoint, "bolt://", "http://", 1)
	endpoint = strings.Replace(endpoint, "neo4j://", "http://", 1)

	// Create the request
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.SetBasicAuth(n.username, n.password)

	// Execute the request
	resp, err := n.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute HTTP request: %w", err)
	}
	defer resp.Body.Close()

	// Read the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Neo4j API error: %s, status code: %d", string(body), resp.StatusCode)
	}

	// Parse the response
	type ResultColumn struct {
		Name string `json:"name"`
	}

	type ResultRow struct {
		Row []interface{} `json:"row"`
	}

	type ResultData struct {
		Columns []ResultColumn `json:"columns"`
		Data    []ResultRow    `json:"data"`
	}

	type TransactionResponse struct {
		Results []ResultData `json:"results"`
	}

	var response TransactionResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Extract the results
	var results []map[string]interface{}
	if len(response.Results) > 0 {
		for _, row := range response.Results[0].Data {
			result := make(map[string]interface{})
			for i, col := range response.Results[0].Columns {
				if i < len(row.Row) {
					result[col.Name] = row.Row[i]
				}
			}
			results = append(results, result)
		}
	}

	return results, nil
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
	query := "MATCH (n {id: $id}) DETACH DELETE n"
	params := map[string]interface{}{
		"id": id,
	}

	_, err := n.Query(ctx, query, params)
	return err
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
