package memory

import (
	"context"
	"io"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/caramba/utils"
)

type Store interface {
	Connect(ctx context.Context, rw io.ReadWriteCloser) error
	Use(ctx context.Context, args map[string]any) string
}

type Query interface {
	Use(ctx context.Context, args map[string]any) string
}

type LongTerm struct {
	stores  []Store
	queries []Query
}

func NewLongTerm() *LongTerm {
	qdrantCollection := viper.GetViper().GetString("tools.qdrant.collection")
	qdrantDimension := viper.GetViper().GetUint64("tools.qdrant.dimension")

	return &LongTerm{
		stores: []Store{
			tools.NewQdrantStore(qdrantCollection, qdrantDimension),
			tools.NewNeo4jStore(),
		},
		queries: []Query{
			tools.NewQdrantQuery(qdrantCollection, qdrantDimension),
			tools.NewNeo4jQuery(),
		},
	}
}

func (longterm *LongTerm) Name() string {
	return "longterm"
}

func (longterm *LongTerm) Description() string {
	return "Long-term memory"
}

func (longterm *LongTerm) GenerateSchema() interface{} {
	return utils.GenerateSchema[*LongTerm]()
}

func (longterm *LongTerm) Initialize() error {
	return nil
}

func (longterm *LongTerm) Connect(ctx context.Context, rw io.ReadWriteCloser) error {
	return nil
}

func (longterm *LongTerm) Use(ctx context.Context, args map[string]any) string {
	return ""
}
