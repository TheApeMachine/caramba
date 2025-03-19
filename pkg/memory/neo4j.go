package memory

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

	neo4j := &Neo4j{
		client: driver,
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("memory.Neo4j.buffer")
			return nil
		}),
	}

	return neo4j
}

func (n4j *Neo4j) Read(p []byte) (n int, err error) {
	errnie.Debug("Neo4j.Read")
	if n4j.buffer == nil {
		return 0, errnie.Error(err)
	}
	return n4j.buffer.Read(p)
}

func (n4j *Neo4j) Write(p []byte) (n int, err error) {
	errnie.Debug("Neo4j.Write")
	if n4j.buffer == nil {
		return 0, errnie.Error(err)
	}
	return n4j.buffer.Write(p)
}

func (n4j *Neo4j) Close() error {
	errnie.Debug("Neo4j.Close")

	if n4j.client != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		if err := n4j.client.Close(ctx); err != nil {
			return errnie.Error(err)
		}
	}

	if n4j.buffer != nil {
		return n4j.buffer.Close()
	}

	return nil
}
