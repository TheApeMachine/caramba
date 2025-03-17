package utils

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
)

func SendEvent(
	writer io.Writer,
	origin string,
	role message.Role,
	content string,
) error {
	msg, err := message.New(
		role,
		"",
		content,
	).Message().Marshal()

	if errnie.Error(err) != nil {
		return err
	}

	evt, err := event.New(
		origin,
		event.MessageEvent,
		event.AssistantRole,
		msg,
	).Message().Marshal()

	if errnie.Error(err) != nil {
		return err
	}

	_, err = writer.Write(evt)

	if errnie.Error(err) != nil {
		return err
	}

	return nil
}
