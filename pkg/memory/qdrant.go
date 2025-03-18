package memory

import (
	"os"
	"strconv"

	sdk "github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Qdrant struct {
	Store
	client     *sdk.Client
	collection string
	buffer     *stream.Buffer
}

func NewQdrant() *Qdrant {
	errnie.Debug("memory.NewQdrant")

	port, err := strconv.Atoi(os.Getenv("QDRANT_PORT"))
	if errnie.Error(err) != nil {
		return nil
	}

	client, err := sdk.NewClient(&sdk.Config{
		Host:   os.Getenv("QDRANT_HOST"),
		Port:   port,
		APIKey: os.Getenv("QDRANT_API_KEY"),
	})

	if err != nil {
		errnie.Error(err)
		return nil
	}

	qdrant := &Qdrant{
		client:     client,
		collection: os.Getenv("QDRANT_COLLECTION"),
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("memory.Qdrant.buffer")
			return nil
		}),
	}

	return qdrant
}

func (q *Qdrant) Read(p []byte) (n int, err error) {
	errnie.Debug("Qdrant.Read")
	if q.buffer == nil {
		return 0, errnie.Error(err)
	}
	return q.buffer.Read(p)
}

func (q *Qdrant) Write(p []byte) (n int, err error) {
	errnie.Debug("Qdrant.Write")
	if q.buffer == nil {
		return 0, errnie.Error(err)
	}
	return q.buffer.Write(p)
}

func (q *Qdrant) Close() error {
	errnie.Debug("Qdrant.Close")
	if q.buffer != nil {
		return q.buffer.Close()
	}
	return nil
}
