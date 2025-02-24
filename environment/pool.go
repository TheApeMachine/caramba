package environment

import "sync"

var once sync.Once
var instance *Pool

type Pool struct {
	Executors []*Executor
}

func NewPool() *Pool {
	once.Do(func() {
		instance = &Pool{
			Executors: make([]*Executor, 0),
		}
	})

	return instance
}

func (p *Pool) AddExecutor(executor *Executor) {
	p.Executors = append(p.Executors, executor)
}

func (p *Pool) RemoveExecutor(executor *Executor) {
	p.Executors = p.Executors[:len(p.Executors)-1]
}

func (p *Pool) GetExecutor(id string) *Executor {
	for _, executor := range p.Executors {
		if executor.Agent.Identity.ID == id {
			return executor
		}
	}
	return nil
}
