package twoface

import (
	"context"
	"slices"
	"strings"
	"sync"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Hub struct {
	ctx       context.Context
	cancel    context.CancelFunc
	wg        *sync.WaitGroup
	router    *Router
	address   string
	agents    []string
	topics    map[string][]string
	tools     map[string][]string
	providers []string
	queue     [][][]byte
	mutex     sync.RWMutex
	muQueue   sync.RWMutex
}

func NewHub(pctx context.Context, wg *sync.WaitGroup) (*Hub, error) {
	addr := viper.GetViper().GetString("settings.hub.address")
	router, err := NewRouter(addr)

	if err != nil {
		return nil, errnie.InternalError(err)
	}

	ctx, cancel := context.WithCancel(pctx)

	return &Hub{
		ctx:       ctx,
		cancel:    cancel,
		wg:        wg,
		router:    router,
		address:   addr,
		agents:    make([]string, 0),
		topics:    make(map[string][]string),
		tools:     make(map[string][]string),
		providers: make([]string, 0),
		queue:     make([][][]byte, 0),
		mutex:     sync.RWMutex{},
		muQueue:   sync.RWMutex{},
	}, nil
}

func (hub *Hub) Start() (err error) {
	errnie.Trace("twoface.hub.Start")

	go func() {
		for {
			select {
			case <-hub.ctx.Done():
				hub.Stop()
				return
			default:
				var msg [][]byte

				if msg, err = hub.router.sock.RecvMessage(); err != nil {
					errnie.Error(errnie.Wrap(err, "failed to receive message"))
					break
				}

				hub.muQueue.Lock()
				hub.queue = append(hub.queue, msg)
				hub.muQueue.Unlock()
			}
		}
	}()

	go func() {
		for {
			select {
			case <-hub.ctx.Done():
				return
			default:
				var msg [][]byte

				hub.muQueue.Lock()
				if len(hub.queue) > 0 {
					msg, hub.queue = hub.queue[0], hub.queue[1:]
					hub.process(msg)
				}
				hub.muQueue.Unlock()
			}
		}
	}()

	return nil
}

func (hub *Hub) Stop() error {
	errnie.Trace("twoface.hub.Stop")
	hub.cancel()
	hub.wg.Done()
	return nil
}

func (hub *Hub) process(msg [][]byte) {
	errnie.Trace("twoface.hub.process")

	artifact := datura.New(
		datura.WithBytes(msg[1]),
	)

	if artifact.HasError() {
		errnie.Error(artifact.Error())
		return
	}

	switch datura.ArtifactRole(artifact.Role()) {
	case datura.ArtifactRoleSubscriber:
		hub.addSubscriber(artifact)
	case datura.ArtifactRolePublisher:
		hub.publish(artifact)
	}
}

func (hub *Hub) addSubscriber(artifact *datura.Artifact) {
	errnie.Trace("twoface.hub.addSubscriber")

	hub.mutex.Lock()
	defer hub.mutex.Unlock()

	switch datura.ArtifactScope(artifact.Scope()) {
	case datura.ArtifactScopeAgent:
		hub.agents = append(hub.agents, errnie.Try(artifact.Issuer()))
	case datura.ArtifactScopeTool:
		hub.tools[errnie.Try(artifact.Issuer())] = strings.Split(
			datura.GetMetaValue[string](artifact, "operations"),
			",",
		)
	case datura.ArtifactScopeProvider:
		hub.providers = append(hub.providers, errnie.Try(artifact.Issuer()))
	case datura.ArtifactScopeTopic:
		hub.topics[errnie.Try(artifact.Issuer())] = strings.Split(
			datura.GetMetaValue[string](artifact, "topics"),
			",",
		)
	}
}

func (hub *Hub) publish(artifact *datura.Artifact) {
	errnie.Trace("twoface.hub.publish")

	hub.mutex.Lock()
	defer hub.mutex.Unlock()

	var ok bool
	topic := datura.GetMetaValue[string](artifact, "topic")
	to := datura.GetMetaValue[string](artifact, "to")

	switch datura.ArtifactRole(artifact.Role()) {
	case datura.ArtifactRoleAssistant:
		for _, agent := range hub.topics[topic] {
			if slices.Contains(hub.agents, agent) {
				ok = true
				hub.router.sock.Write(artifact.Bytes())
			}
		}

		if slices.Contains(hub.agents, to) {
			ok = true
			hub.router.sock.Write(artifact.Bytes())
		}
	case datura.ArtifactRoleTool:
		tool := datura.GetMetaValue[string](artifact, "tool")
		operation := datura.GetMetaValue[string](artifact, "operation")

		for _, op := range hub.tools[tool] {
			if op == operation {
				ok = true
				hub.router.sock.Write(artifact.Bytes())
			}
		}
	case datura.ArtifactRoleUser:
		provider := datura.GetMetaValue[string](artifact, "provider")

		switch datura.ArtifactScope(artifact.Scope()) {
		case datura.ArtifactScopeGeneration:
			if slices.Contains(hub.providers, provider) {
				ok = true
				hub.router.sock.Write(artifact.Bytes())
			}
		}
	}

	if !ok {
		errnie.Warn("no subscriber", "topic", topic, "to", to)

		hub.muQueue.Lock()
		hub.queue = append(hub.queue, [][]byte{
			artifact.Bytes(),
		})
		hub.muQueue.Unlock()
	}
}
