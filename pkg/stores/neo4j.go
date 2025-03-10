package stores

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	sdk "github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Neo4jData struct {
	Event *core.Event `json:"events"`
}

type Neo4j struct {
	*Neo4jData
	client sdk.DriverWithContext
	enc    *json.Encoder
	dec    *json.Decoder
	in     *bytes.Buffer
	out    *bytes.Buffer
}

func NewNeo4j(collection string) *Neo4j {
	errnie.Debug("NewNeo4j")

	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	driver, err := sdk.NewDriverWithContext(
		os.Getenv("NEO4J_URL"),
		sdk.BasicAuth(
			os.Getenv("NEO4J_USERNAME"),
			os.Getenv("NEO4J_PASSWORD"),
			"",
		),
	)
	if err != nil {
		return nil
	}

	neo4j := &Neo4j{
		Neo4jData: &Neo4jData{},
		client:    driver,
		enc:       json.NewEncoder(out),
		dec:       json.NewDecoder(in),
		in:        in,
		out:       out,
	}

	return neo4j
}

func (neo4j *Neo4j) Read(p []byte) (n int, err error) {
	errnie.Debug("Neo4j.Read")

	if neo4j.out.Len() == 0 {
		return 0, io.EOF
	}

	return neo4j.out.Read(p)
}

// executeQuery performs Neo4j queries based on tool calls and updates the output buffer
// This should be called from Write or other methods that modify data, not from Read
func (neo4j *Neo4j) executeQuery() error {
	var results strings.Builder
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	for _, toolCall := range neo4j.Event.ToolCalls {
		if toolCall.ToolName == "neo4j" {
			session := neo4j.client.NewSession(ctx, sdk.SessionConfig{
				DatabaseName: "neo4j",
				AccessMode:   sdk.AccessModeWrite,
			})

			defer session.Close(ctx)

			// Simple query to find relationships
			result, err := session.Run(
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
					"term": toolCall.Arguments["term"].(string),
				},
			)

			if err != nil {
				return err
			}

			for result.Next(ctx) {
				record := result.Record()
				asmap := record.AsMap()

				results.WriteString(
					fmt.Sprintf("%v:%v -[%v]-> %v:%v\n",
						asmap["sourceLabel"],
						asmap["source"],
						asmap["relationship"],
						asmap["targetLabel"],
						asmap["target"],
					),
				)
			}

			if err := result.Err(); err != nil {
				return err
			}

			if results.Len() == 0 {
				results.WriteString(fmt.Sprintf("No relationships found for: %s\n", toolCall.Arguments["term"].(string)))
				return nil
			}

			// Check if there's a query in the arguments
			if qry, ok := toolCall.Arguments["query"].(string); ok {
				result, err := session.Run(ctx, qry, nil)

				if err != nil {
					return err
				}

				// Format the results
				for result.Next(ctx) {
					record := result.Record()
					results.WriteString(fmt.Sprintf("%v\n", record.AsMap()))
				}

				if err := result.Err(); err != nil {
					return err
				}
			}
		}
	}

	neo4j.Neo4jData.Event = core.NewEvent(
		core.NewMessage("assistant", "neo4j", ""),
		nil,
	)

	return errnie.NewErrIO(neo4j.enc.Encode(neo4j.Neo4jData))
}

func (neo4j *Neo4j) Write(p []byte) (n int, err error) {
	errnie.Debug("Neo4j.Write", "p", string(p))

	// Reset the output buffer whenever we write new data
	if neo4j.out.Len() > 0 {
		neo4j.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = neo4j.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf core.Event
	if decErr := neo4j.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		neo4j.Event = &buf

		// Execute any queries based on tool calls
		if neo4j.Event != nil && len(neo4j.Event.ToolCalls) > 0 {
			if err := neo4j.executeQuery(); err != nil {
				return n, err
			}
		} else {
			// No tool calls, just re-encode
			if err = errnie.NewErrIO(neo4j.enc.Encode(neo4j.Neo4jData)); err != nil {
				return n, err
			}
		}
	}

	return n, nil
}

func (neo4j *Neo4j) Close() (err error) {
	errnie.Debug("Neo4j.Close")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	neo4j.client.Close(ctx)
	return nil
}
