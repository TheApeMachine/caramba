package memory

import "github.com/theapemachine/caramba/pkg/errnie"

type Qdrant struct {
	Store
}

func NewQdrant() *Qdrant {
	return &Qdrant{
		Store: Store{},
	}
}

func (q *Qdrant) Read(p []byte) (n int, err error) {
	errnie.Debug("Qdrant.Read")
	return 0, nil
}

func (q *Qdrant) Write(p []byte) (n int, err error) {
	errnie.Debug("Qdrant.Write")
	return 0, nil
}

func (q *Qdrant) Close() error {
	errnie.Debug("Qdrant.Close")
	return nil
}
