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
var executeTestSetup = func() func(t *testing.T) *rod.Browser {
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

func TestExecuteScript(t *testing.T) {
	Convey("Given a Tool instance", t, func() {
		// Test parameter validation first (doesn't require an actual browser)
		Convey("Parameter validation tests", func() {
			tool := &Tool{}

			Convey("When executeScript is called without a script", func() {
				args := map[string]any{
					"url": "https://example.com",
				}
				_, err := tool.executeScript(context.Background(), args)

				Convey("Then an error should be returned", func() {
					So(err, ShouldNotBeNil)
					So(err.Error(), ShouldContainSubstring, "script must be a string")
				})
			})

			Convey("When executeScript is called with an empty script", func() {
				args := map[string]any{
					"script": "",
					"url":    "https://example.com",
				}
				_, err := tool.executeScript(context.Background(), args)

				Convey("Then an error should be returned", func() {
					So(err, ShouldNotBeNil)
					So(err.Error(), ShouldContainSubstring, "script cannot be empty")
				})
			})

			Convey("When executeScript is called with a script but no URL", func() {
				args := map[string]any{
					"script": "() => { return document.title; }",
				}
				_, err := tool.executeScript(context.Background(), args)

				Convey("Then an error should be returned", func() {
					So(err, ShouldNotBeNil)
					So(err.Error(), ShouldContainSubstring, "url")
				})
			})
		})

		// Skip integration tests if running in short mode
		if testing.Short() {
			t.Skip("Skipping integration tests in short mode")
		}

		// Real browser tests
		Convey("Integration tests with real browser", func() {
			browser := executeTestSetup(t)
			tool := &Tool{
				browser: browser,
				timeout: 5 * time.Second,
			}

			Convey("When executeScript is called with a valid script and URL", func() {
				args := map[string]any{
					"script": "() => { return document.title; }",
					"url":    "https://example.com",
				}
				resultInterface, err := tool.executeScript(context.Background(), args)

				Convey("Then no error should be returned", func() {
					So(err, ShouldBeNil)
					So(resultInterface, ShouldNotBeNil)

					// Type assertion for the result map
					result, ok := resultInterface.(map[string]interface{})
					So(ok, ShouldBeTrue)

					// Check that result contains the script result
					scriptResult, ok := result["result"]
					So(ok, ShouldBeTrue)
					So(scriptResult, ShouldEqual, "Example Domain")
				})
			})

			Convey("When executeScript is called with a script to extract page elements", func() {
				args := map[string]any{
					"script": "() => { return document.querySelector('h1').textContent; }",
					"url":    "https://example.com",
				}
				resultInterface, err := tool.executeScript(context.Background(), args)

				Convey("Then no error should be returned", func() {
					So(err, ShouldBeNil)
					So(resultInterface, ShouldNotBeNil)

					// Type assertion for the result map
					result, ok := resultInterface.(map[string]any)
					So(ok, ShouldBeTrue)

					// Check that result contains the script result
					scriptResult, ok := result["result"]
					So(ok, ShouldBeTrue)
					So(scriptResult, ShouldEqual, "Example Domain")
				})
			})
		})
	})
}
