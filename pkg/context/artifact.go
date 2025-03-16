package context

import (
	"capnproto.org/go/capnp/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/message"
)

func root() (*Artifact, error) {
	arena := capnp.SingleSegment(nil)

	_, seg, err := capnp.NewMessage(arena)
	if errnie.Error(err) != nil {
		return nil, err
	}

	artfct, err := NewRootArtifact(seg)
	if errnie.Error(err) != nil {
		return nil, err
	}

	return &artfct, nil
}

/*
New creates a new artifact with the given origin, role, scope, and data.
*/
func New(
	model string,
	messages []*Message,
	tools []*Tool,
	process []byte,
	temperature float64,
	topP float64,
	topK int,
	presencePenalty float64,
	frequencyPenalty float64,
	maxTokens int,
	stream bool,
) *Artifact {
	var (
		err      error
		artifact *Artifact
	)

	if artifact, err = root(); errnie.Error(err) != nil {
		return nil
	}

	artifact.SetId(uuid.New().String())

	// Error handling: if setting any required field fails, return Empty()
	if err := artifact.SetModel(model); err != nil {
		errnie.Error(err)
		return nil
	}

	// Add an empty message list.
	msgs, err := NewMessage_List(artifact.Segment(), 0)

	if err != nil {
		errnie.Error(err)
		return nil
	}

	if err := artifact.SetMessages(msgs); err != nil {
		errnie.Error(err)
		return nil
	}

	// Add an empty tool list.
	toolList, err := NewTool_List(artifact.Segment(), 0)
	if err != nil {
		errnie.Error(err)
		return nil
	}

	if err := artifact.SetTools(toolList); err != nil {
		errnie.Error(err)
		return nil
	}

	if err := artifact.SetProcess(process); err != nil {
		errnie.Error(err)
		return nil
	}

	artifact.SetTemperature(temperature)
	artifact.SetTopP(topP)
	artifact.SetTopK(float64(topK))
	artifact.SetPresencePenalty(presencePenalty)
	artifact.SetFrequencyPenalty(frequencyPenalty)
	artifact.SetMaxTokens(int32(maxTokens))
	artifact.SetStream(stream)

	return artifact
}

/*
AddMessage adds a message to the artifact.
*/
func (a *Artifact) AddMessage(msgArtifact *message.Artifact) error {
	// Get current messages
	msgs, err := a.Messages()
	if err != nil {
		return err
	}

	// Create new message list with size + 1
	newMsgs, err := NewMessage_List(a.Segment(), int32(msgs.Len()+1))
	if err != nil {
		return err
	}

	// Copy existing messages
	for i := 0; i < msgs.Len(); i++ {
		if err := newMsgs.Set(i, msgs.At(i)); err != nil {
			return err
		}
	}

	// Create new context.Message
	newMsg, err := NewMessage(a.Segment())
	if err != nil {
		return err
	}

	// Copy fields from message.Artifact to context.Message
	id, err := msgArtifact.Id()
	if err == nil {
		newMsg.SetId(id)
	}

	role, err := msgArtifact.Role()
	if err == nil {
		newMsg.SetRole(role)
	}

	name, err := msgArtifact.Name()
	if err == nil {
		newMsg.SetName(name)
	}

	content, err := msgArtifact.Content()
	if err == nil {
		newMsg.SetContent(content)
	}

	// Add new message at the end
	if err := newMsgs.Set(msgs.Len(), newMsg); err != nil {
		return err
	}

	// Set the new message list
	return a.SetMessages(newMsgs)
}
