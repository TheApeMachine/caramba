package stores

import (
	"context"
	"os"
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

func NewNeo4j(collection string) *Neo4j {
	errnie.Debug("stores.NewNeo4j")

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
		client: driver,
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("stores.Neo4j.buffer")
			return nil
		}),
	}

	return neo4j
}

func (neo4j *Neo4j) Read(p []byte) (n int, err error) {
	errnie.Debug("stores.Neo4j.Read")
	return neo4j.buffer.Read(p)
}

func (neo4j *Neo4j) Write(p []byte) (n int, err error) {
	errnie.Debug("stores.Neo4j.Write")
	return neo4j.buffer.Write(p)
}

func (neo4j *Neo4j) Close() (err error) {
	errnie.Debug("stores.Neo4j.Close")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	neo4j.client.Close(ctx)
	return nil
}
