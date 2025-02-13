package system

import (
	"sync"

	sdk "github.com/openai/openai-go"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/errnie"
)

type Envelope struct {
	From    string
	To      string
	Payload *provider.StructuredParams
}

type Channel struct {
	identifier string
	route      chan *Envelope
	i          chan *Envelope
	o          chan *provider.StructuredParams
}

func (channel *Channel) Start() {
	errnie.Info("channel", "identifier", channel.identifier, "operation", "start")

	go func() {
		for {
			select {
			case message := <-channel.i:
				errnie.Info("send", "sender", channel.identifier)
				channel.route <- message
			}
		}
	}()
}

type Queue struct {
	channels map[string]*Channel
	topics   map[string][]string
	route    chan *Envelope
}

var queueOnce sync.Once
var queueInstance *Queue

func NewQueue() *Queue {
	queueOnce.Do(func() {
		queueInstance = &Queue{
			channels: make(map[string]*Channel),
			route:    make(chan *Envelope),
		}
	})

	return queueInstance
}

func (queue *Queue) Subscribe(topic, identifier string) {
	if queue.topics == nil {
		queue.topics = make(map[string][]string)
	}

	if _, ok := queue.topics[topic]; !ok {
		queue.topics[topic] = []string{identifier}
		return
	}

	queue.topics[topic] = append(queue.topics[topic], identifier)
}

func (queue *Queue) Claim(
	identifier string,
) (chan *Envelope, chan *provider.StructuredParams) {
	errnie.Info("claim", "identifier", identifier)

	i := make(chan *Envelope)
	o := make(chan *provider.StructuredParams)

	queue.channels[identifier] = &Channel{
		identifier: identifier,
		route:      queue.route,
		i:          i,
		o:          o,
	}

	queue.channels[identifier].Start()

	return i, o
}

func (queue *Queue) Ingress(message *Envelope) {
	errnie.Info("ingress")

	if message.To == "" {
		for _, channel := range queue.channels {
			channel.o <- message.Payload
		}
		return
	}

	if channel, ok := queue.channels[message.To]; ok {
		channel.o <- message.Payload
	}
}

func (queue *Queue) Start() {
	errnie.Info("queue", "operation", "start")

	go func() {
		for {
			select {
			case message := <-queue.route:
				if _, ok := queue.channels[message.To]; !ok {
					if message.To == "" {
						// Broadcasting to all.
						for identifier, channel := range queue.channels {
							// Do not broadcast to sender.
							if identifier == message.From {
								continue
							}

							channel.o <- message.Payload
						}
					}

					message.Payload.Messages = append(
						message.Payload.Messages,
						sdk.AssistantMessage("return to sender, address unknown"),
					)

					queue.channels[message.From].o <- message.Payload
					continue
				}

				queue.channels[message.To].o <- message.Payload
			}
		}
	}()
}
