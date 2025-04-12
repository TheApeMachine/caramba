package neo4j

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/gofiber/fiber/v3"
	sdk "github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type N4jQuery struct {
	Cypher   string
	Keywords map[string]string
	Params   map[string]any
}

type Neo4j struct {
	client sdk.DriverWithContext
}

func NewNeo4j(collection string) *Neo4j {
	errnie.Debug("NewNeo4j")

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
	}

	return neo4j
}

func (neo4j *Neo4j) findRelationships(ctx fiber.Ctx, session sdk.SessionWithContext, keyword string) (string, error) {
	var results strings.Builder

	result, err := session.Run(
		ctx.Context(),
		`
		MATCH p=(a)-[r]->(b)
		WHERE a.name CONTAINS $term OR b.name CONTAINS $term
		RETURN a.name as source, labels(a)[0] as sourceLabel,
			type(r) as relationship,
			b.name as target, labels(b)[0] as targetLabel
		LIMIT 20
		`,
		map[string]any{
			"term": keyword,
		},
	)

	if err != nil {
		return "", errnie.New(errnie.WithError(err))
	}

	for result.Next(ctx.Context()) {
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
		return "", errnie.New(errnie.WithError(err))
	}

	if results.Len() == 0 {
		return fmt.Sprintf("No relationships found for: %s\n", keyword), nil
	}

	return results.String(), nil
}

func (neo4j *Neo4j) executeQuery(ctx fiber.Ctx, query N4jQuery) (string, error) {
	session := neo4j.client.NewSession(ctx.Context(), sdk.SessionConfig{
		DatabaseName: "neo4j",
		AccessMode:   sdk.AccessModeWrite,
	})
	defer session.Close(ctx.Context())

	var results strings.Builder

	for _, keyword := range query.Keywords {
		result, err := neo4j.findRelationships(ctx, session, keyword)
		if err != nil {
			return "", err
		}
		results.WriteString(result)
	}

	if query.Cypher != "" {
		result, err := session.Run(ctx.Context(), query.Cypher, query.Params)
		if err != nil {
			return "", errnie.New(errnie.WithError(err))
		}

		for result.Next(ctx.Context()) {
			results.WriteString(fmt.Sprintf("%v\n", result.Record().AsMap()))
		}

		if err := result.Err(); err != nil {
			return "", errnie.New(errnie.WithError(err))
		}
	}

	return results.String(), nil
}

func (neo4j *Neo4j) Get() (err error) {
	neo4j.client.Close(context.Background())
	return nil
}

func (neo4j *Neo4j) Put(ctx fiber.Ctx, n4jQuery N4jQuery) (string, error) {
	results, err := neo4j.executeQuery(ctx, n4jQuery)
	if err != nil {
		return "", err
	}
	return results, nil
}
