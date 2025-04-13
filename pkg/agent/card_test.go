package agent

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewCard(t *testing.T) {
	Convey("Given a card", t, func() {
		card := NewCard()

		So(card, ShouldNotBeNil)
	})
}

func TestFromConfig(t *testing.T) {
	Convey("Given a card", t, func() {
		card := FromConfig("test")

		So(card, ShouldNotBeNil)
	})
}
