package service

import (
	"context"
	"sync"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/twoface"
)

type Component interface {
	ID() string
	Name() string
	HandleMessage(*datura.Artifact) *datura.Artifact
}

type ComponentService struct {
	ctx           context.Context
	cancel        context.CancelFunc
	wg            *sync.WaitGroup
	stopChan      chan struct{}
	hubAddr       string
	transport     *twoface.Transport
	name          string
	component     Component
	subscriptions []string
}

func NewComponentService(
	pctx context.Context,
	wg *sync.WaitGroup,
	name string,
	component Component,
	subscriptions []string,
) (*ComponentService, error) {
	errnie.Trace("NewComponentService")

	hubAddr := viper.GetViper().GetString("settings.hub.address")

	transport, err := twoface.NewTransport(name)

	if err != nil {
		return nil, errnie.InternalError(err)
	}

	ctx, cancel := context.WithCancel(pctx)

	return &ComponentService{
		ctx:           ctx,
		cancel:        cancel,
		wg:            wg,
		stopChan:      make(chan struct{}),
		hubAddr:       hubAddr,
		transport:     transport,
		name:          name,
		component:     component,
		subscriptions: subscriptions,
	}, nil
}

func (srv *ComponentService) Start() (err error) {
	errnie.Trace("ComponentService.Start")

	if len(srv.subscriptions) > 0 {
		errnie.Debug("ComponentService.Start", "subscriptions", srv.subscriptions)

		if err = srv.transport.Subscribe(srv.subscriptions); err != nil {
			return errnie.InternalError(err)
		}
	}

	var msgFrames [][]byte

	go func() {
		for {
			select {
			case <-srv.ctx.Done():
				return
			case <-srv.stopChan:
				srv.Stop()
				return
			default:
				if msgFrames, err = srv.transport.Recv(); err != nil {
					errnie.InternalError(err)
				}

				artifact := datura.New(
					datura.WithBytes(msgFrames[1]),
				)

				if artifact.HasError() {
					errnie.InternalError(artifact)
					continue
				}

				srv.transport.Publish(
					srv.component.HandleMessage(artifact),
				)
			}
		}
	}()

	return nil
}

func (srv *ComponentService) Stop() error {
	errnie.Trace("ComponentService.Stop")

	srv.cancel()
	close(srv.stopChan)
	srv.wg.Done()

	return nil
}
