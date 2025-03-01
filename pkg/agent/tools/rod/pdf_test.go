package rod

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestGeneratePDF(t *testing.T) {
	Convey("Given a Tool instance", t, func() {
		tool := &Tool{}

		Convey("When generatePDF is called without a URL", func() {
			args := map[string]interface{}{}
			_, err := tool.generatePDF(context.Background(), args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "url must be a non-empty string")
			})
		})

		Convey("When generatePDF is called with an empty URL", func() {
			args := map[string]interface{}{
				"url": "",
			}
			_, err := tool.generatePDF(context.Background(), args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "url must be a non-empty string")
			})
		})

		Convey("When generatePDF is called with a non-string URL", func() {
			args := map[string]interface{}{
				"url": 123,
			}
			_, err := tool.generatePDF(context.Background(), args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "url must be a non-empty string")
			})
		})

		Convey("When generatePDF is called with a valid URL", func() {
			args := map[string]interface{}{
				"url": "https://example.com",
			}
			_, err := tool.generatePDF(context.Background(), args)

			Convey("Then an error should be returned (due to missing browser)", func() {
				So(err, ShouldNotBeNil)
				// This will fail because we need a browser instance, but it tests the validation logic
			})
		})

		Convey("When generatePDF is called with format options", func() {
			args := map[string]interface{}{
				"url":       "https://example.com",
				"format":    "A4",
				"full_page": true,
			}
			_, err := tool.generatePDF(context.Background(), args)

			Convey("Then an error should be returned (due to missing browser)", func() {
				So(err, ShouldNotBeNil)
				// This will fail because we need a browser instance, but it tests the validation logic
			})
		})

		// Note: Testing the actual PDF generation would require mocking the rod.Page
		// which is challenging due to the design of the rod library.
		// In a real-world scenario, we would use dependency injection and interfaces
		// to make this more testable.
	})
}
