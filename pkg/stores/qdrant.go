package stores

import (
	"os"
	"strconv"

	sdk "github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Qdrant struct {
	client     *sdk.Client
	collection string
	buffer     *stream.Buffer
}

func NewQdrant(collection string) *Qdrant {
	errnie.Debug("stores.NewQdrant")

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
		return nil
	}

	return &Qdrant{
		client:     client,
		collection: collection,
	}
}

func (qdrant *Qdrant) Read(p []byte) (n int, err error) {
	errnie.Debug("stores.Qdrant.Read")
	return qdrant.buffer.Read(p)
}

func (qdrant *Qdrant) Write(p []byte) (n int, err error) {
	errnie.Debug("stores.Qdrant.Write")
	return qdrant.buffer.Write(p)
}

func (qdrant *Qdrant) Close() (err error) {
	errnie.Debug("stores.Qdrant.Close")
	return qdrant.buffer.Close()
}
