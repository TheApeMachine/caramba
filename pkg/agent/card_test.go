package agent

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestNewCard(t *testing.T) {
	Convey("Given a card", t, func() {
		card := NewCard()

		So(card, ShouldNotBeNil)
	})
}

func TestFromConfig(t *testing.T) {
	v := viper.New()
	v.AddConfigPath("../../cmd/cfg")
	v.SetConfigName("config")
	v.SetConfigType("yml")

	if err := v.ReadInConfig(); err != nil {
		t.Fatalf("Failed to read config: %v", err)
	}

	Convey("Given a card", t, func() {
		card := FromConfig("ui")

		So(card, ShouldNotBeNil)
	})
}
