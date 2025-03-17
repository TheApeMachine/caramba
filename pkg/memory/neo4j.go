package memory

import "github.com/theapemachine/caramba/pkg/errnie"

type Neo4j struct {
	Store
}

func NewNeo4j() *Neo4j {
	return &Neo4j{
		Store: Store{},
	}
}

func (n4j *Neo4j) Read(p []byte) (n int, err error) {
	errnie.Debug("Neo4j.Read")
	return 0, nil
}

func (n4j *Neo4j) Write(p []byte) (n int, err error) {
	errnie.Debug("Neo4j.Write")
	return 0, nil
}

func (n4j *Neo4j) Close() error {
	errnie.Debug("Neo4j.Close")
	return nil
}
