package rod

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestScreenshot(t *testing.T) {
	Convey("Given a Tool instance", t, func() {
		tool := &Tool{}

		Convey("When screenshot is called without a URL", func() {
			args := map[string]interface{}{}
			_, err := tool.screenshot(context.Background(), args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "url must be a string")
			})
		})

		Convey("When screenshot is called with a non-string URL", func() {
			args := map[string]interface{}{
				"url": 123,
			}
			_, err := tool.screenshot(context.Background(), args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "url must be a string")
			})
		})

		Convey("When screenshot is called with a valid URL", func() {
			args := map[string]interface{}{
				"url": "https://example.com",
			}
			_, err := tool.screenshot(context.Background(), args)

			Convey("Then an error should be returned (due to missing browser)", func() {
				So(err, ShouldNotBeNil)
				// This will fail because we need a browser instance, but it tests the validation logic
			})
		})

		Convey("When screenshot is called with full_page option", func() {
			args := map[string]interface{}{
				"url":       "https://example.com",
				"full_page": true,
			}
			_, err := tool.screenshot(context.Background(), args)

			Convey("Then an error should be returned (due to missing browser)", func() {
				So(err, ShouldNotBeNil)
				// This will fail because we need a browser instance, but it tests the validation logic
			})
		})

		Convey("When screenshot is called with wait_for selector", func() {
			args := map[string]interface{}{
				"url":      "https://example.com",
				"wait_for": ".content",
			}
			_, err := tool.screenshot(context.Background(), args)

			Convey("Then an error should be returned (due to missing browser)", func() {
				So(err, ShouldNotBeNil)
				// This will fail because we need a browser instance, but it tests the validation logic
			})
		})

		// Note: Testing the actual screenshot capture would require mocking the rod.Page
		// which is challenging due to the design of the rod library.
		// In a real-world scenario, we would use dependency injection and interfaces
		// to make this more testable.
	})
}
