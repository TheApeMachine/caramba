package rod

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestExtractContent(t *testing.T) {
	Convey("Given a Tool instance", t, func() {
		tool := &Tool{}

		Convey("When extractContent is called without a URL", func() {
			args := map[string]interface{}{
				"selector": ".content",
			}
			_, err := tool.extractContent(context.Background(), args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "url must be a non-empty string")
			})
		})

		Convey("When extractContent is called with an empty URL", func() {
			args := map[string]interface{}{
				"url":      "",
				"selector": ".content",
			}
			_, err := tool.extractContent(context.Background(), args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "url must be a non-empty string")
			})
		})

		Convey("When extractContent is called without a selector", func() {
			args := map[string]interface{}{
				"url": "https://example.com",
			}
			_, err := tool.extractContent(context.Background(), args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "selector must be a non-empty string")
			})
		})

		Convey("When extractContent is called with an empty selector", func() {
			args := map[string]interface{}{
				"url":      "https://example.com",
				"selector": "",
			}
			_, err := tool.extractContent(context.Background(), args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "selector must be a non-empty string")
			})
		})

		Convey("When extractContent is called with valid URL and selector", func() {
			args := map[string]interface{}{
				"url":      "https://example.com",
				"selector": ".content",
			}
			_, err := tool.extractContent(context.Background(), args)

			Convey("Then an error should be returned (due to missing browser)", func() {
				So(err, ShouldNotBeNil)
				// This will fail because we need a browser instance, but it tests the validation logic
			})
		})

		// Note: Testing the actual content extraction would require mocking the rod.Page
		// which is challenging due to the design of the rod library.
		// In a real-world scenario, we would use dependency injection and interfaces
		// to make this more testable.
	})
}
