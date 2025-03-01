package rod

import (
	"context"
	"testing"
	"time"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	. "github.com/smartystreets/goconvey/convey"
)

// If testSetup is not already defined in another file that's imported
var navigateTestSetup = func() func(t *testing.T) *rod.Browser {
	// Create a launcher with leakless mode
	l := launcher.New().Leakless(true).Headless(true)

	// Launch the browser and connect
	browser := rod.New().ControlURL(l.MustLaunch()).MustConnect()
	browser.Timeout(5 * time.Second)

	return func(t *testing.T) *rod.Browser {
		t.Parallel() // run each test concurrently
		return browser
	}
}()

func TestNavigate(t *testing.T) {
	Convey("Given a Tool instance", t, func() {
		// Test parameter validation first (doesn't require an actual browser)
		Convey("Parameter validation tests", func() {
			tool := &Tool{}

			Convey("When navigate is called without a URL", func() {
				args := map[string]interface{}{}
				_, err := tool.navigate(context.Background(), args)

				Convey("Then an error should be returned", func() {
					So(err, ShouldNotBeNil)
					So(err.Error(), ShouldContainSubstring, "url must be a string")
				})
			})

			Convey("When navigate is called with a non-string URL", func() {
				args := map[string]interface{}{
					"url": 123,
				}
				_, err := tool.navigate(context.Background(), args)

				Convey("Then an error should be returned", func() {
					So(err, ShouldNotBeNil)
					So(err.Error(), ShouldContainSubstring, "url must be a string")
				})
			})
		})

		// Skip integration tests if running in short mode
		if testing.Short() {
			t.Skip("Skipping integration tests in short mode")
		}

		// Real browser tests
		Convey("Integration tests with real browser", func() {
			browser := navigateTestSetup(t)
			tool := &Tool{
				browser: browser,
				timeout: 5 * time.Second,
			}

			Convey("When navigate is called with a valid URL", func() {
				args := map[string]interface{}{
					"url": "https://example.com",
				}
				resultInterface, err := tool.navigate(context.Background(), args)

				Convey("Then no error should be returned", func() {
					So(err, ShouldBeNil)
					So(resultInterface, ShouldNotBeNil)

					// Type assertion for the result map
					result, ok := resultInterface.(map[string]interface{})
					So(ok, ShouldBeTrue)

					// Check that result contains the expected fields
					_, ok = result["page"]
					So(ok, ShouldBeTrue)

					title, ok := result["title"]
					So(ok, ShouldBeTrue)
					So(title, ShouldNotBeEmpty)
				})
			})

			Convey("When navigate is called with a URL and wait_for selector", func() {
				// Example.com has an h1 element
				args := map[string]interface{}{
					"url":      "https://example.com",
					"wait_for": "h1",
				}
				resultInterface, err := tool.navigate(context.Background(), args)

				Convey("Then no error should be returned", func() {
					So(err, ShouldBeNil)
					So(resultInterface, ShouldNotBeNil)

					// Type assertion for the result map
					result, ok := resultInterface.(map[string]interface{})
					So(ok, ShouldBeTrue)

					// Check that the page was loaded and the selector was found
					_, ok = result["page"]
					So(ok, ShouldBeTrue)

					title, ok := result["title"]
					So(ok, ShouldBeTrue)
					So(title, ShouldEqual, "Example Domain")
				})
			})

			Convey("When navigate is called with an invalid URL", func() {
				args := map[string]interface{}{
					"url": "invalid-url",
				}
				_, err := tool.navigate(context.Background(), args)

				Convey("Then an error should be returned", func() {
					So(err, ShouldNotBeNil)
				})
			})
		})
	})
}
