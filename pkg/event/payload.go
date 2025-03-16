package event

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/message"
)

func (artifact *Artifact) ToContext(ctx *context.Artifact) (err error) {
	typ, err := artifact.Type()

	if errnie.Error(err) != nil {
		return nil
	}

	switch typ {
	case string(MessageEvent):
		msg := &message.Artifact{}
		payload, err := artifact.Payload()

		if errnie.Error(err) != nil {
			return nil
		}

		if _, err = msg.Write(payload); errnie.Error(err) != nil {
			return err
		}

		messages, err := ctx.Messages()

		if errnie.Error(err) != nil {
			return err
		}

		newMessages, err := context.NewMessage_List(ctx.Segment(), int32(messages.Len()+1))
		if errnie.Error(err) != nil {
			return err
		}

		// Copy existing messages to the new segment
		for i := 0; i < messages.Len(); i++ {
			oldMsg := messages.At(i)
			newMsg, err := context.NewMessage(ctx.Segment())
			if errnie.Error(err) != nil {
				return err
			}

			// Copy each field individually
			if role, err := oldMsg.Role(); err == nil {
				newMsg.SetRole(role)
			}
			if name, err := oldMsg.Name(); err == nil {
				newMsg.SetName(name)
			}
			if content, err := oldMsg.Content(); err == nil {
				newMsg.SetContent(content)
			}
			if id, err := oldMsg.Id(); err == nil {
				newMsg.SetId(id)
			}

			if err = newMessages.Set(i, newMsg); errnie.Error(err) != nil {
				return err
			}
		}

		// Add the new message
		newMsg, err := context.NewMessage(ctx.Segment())
		if errnie.Error(err) != nil {
			return err
		}

		role, err := msg.Role()
		if errnie.Error(err) != nil {
			return err
		}

		// Validate the role and use a default if invalid
		switch role {
		case "system", "user", "assistant", "tool":
			// Valid role, use it as is
		default:
			errnie.Error("invalid role, using assistant role", "role", role)
			role = "assistant"
		}

		if err = newMsg.SetRole(role); errnie.Error(err) != nil {
			return err
		}

		name, err := msg.Name()
		if errnie.Error(err) != nil {
			return err
		}

		if err = newMsg.SetName(name); errnie.Error(err) != nil {
			return err
		}

		content, err := msg.Content()
		if errnie.Error(err) != nil {
			return err
		}

		if err = newMsg.SetContent(content); errnie.Error(err) != nil {
			return err
		}

		if err = newMessages.Set(messages.Len(), newMsg); errnie.Error(err) != nil {
			return err
		}

		if err = ctx.SetMessages(newMessages); errnie.Error(err) != nil {
			return err
		}
	case string(ContextEvent):
		payload, err := artifact.Payload()

		if errnie.Error(err) != nil {
			return nil
		}

		ctx.Write(payload)
	default:
		typ, err := artifact.Type()

		if errnie.Error(err) != nil {
			return err
		}

		return fmt.Errorf("unknown event type: %s", typ)
	}

	return nil
}
