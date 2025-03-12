package provider

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestNewOpenAIProvider tests the initialization of OpenAIProvider
func TestNewOpenAIProvider(t *testing.T) {
	Convey("Given environment variables for API keys are set", t, func() {
		apiKey := "test-api-key"
		endpoint := "https://api.openai.com/v1"

		Convey("When creating a new OpenAI provider", func() {
			provider := NewOpenAIProvider(apiKey, endpoint)

			Convey("Then it should be properly initialized", func() {
				So(provider, ShouldNotBeNil)
				So(provider.ProviderData, ShouldNotBeNil)
				So(provider.client, ShouldNotBeNil)
			})
		})
	})
}
