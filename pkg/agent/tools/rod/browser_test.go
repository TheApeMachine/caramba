package rod

import (
	"testing"
	"time"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	. "github.com/smartystreets/goconvey/convey"
)

// testSetup creates a shared browser instance and returns a setup function for tests
var testSetup = func() func(t *testing.T) *rod.Browser {
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

// createTestTool creates a Tool with the provided browser
func createTestTool(browser *rod.Browser) *Tool {
	return &Tool{
		browser: browser,
		timeout: 5 * time.Second, // Short timeout for tests
		hub:     nil,             // Hub can be nil for tests that don't use it
	}
}

func TestNewTool(t *testing.T) {
	Convey("Given a need to create a new Rod Browser Tool", t, func() {
		Convey("When New() is called", func() {
			// This test is still challenging since we're creating a new browser
			// Skip test if running in CI/headless environments
			if testing.Short() {
				t.Skip("Skipping test in short mode")
			}

			tool := New()
			defer tool.Close() // Ensure we close the browser

			Convey("Then a new Tool instance should be returned", func() {
				So(tool, ShouldNotBeNil)
				So(tool.browser, ShouldNotBeNil)
				So(tool.timeout.Seconds(), ShouldEqual, 30)
				So(tool.hub, ShouldNotBeNil)
			})
		})
	})
}

func TestToolName(t *testing.T) {
	Convey("Given a Rod Browser Tool instance", t, func() {
		// Get the shared browser instance
		browser := testSetup(t)
		tool := createTestTool(browser)

		Convey("When Name() is called", func() {
			name := tool.Name()

			Convey("Then the correct name should be returned", func() {
				So(name, ShouldEqual, "rod-browser")
			})
		})
	})
}

func TestToolDescription(t *testing.T) {
	Convey("Given a Rod Browser Tool instance", t, func() {
		browser := testSetup(t)
		tool := createTestTool(browser)

		Convey("When Description() is called", func() {
			desc := tool.Description()

			Convey("Then a non-empty description should be returned", func() {
				So(desc, ShouldNotBeEmpty)
				So(desc, ShouldContainSubstring, "interact")
				So(desc, ShouldContainSubstring, "Go Rod")
			})
		})
	})
}

func TestToolSchema(t *testing.T) {
	Convey("Given a Rod Browser Tool instance", t, func() {
		browser := testSetup(t)
		tool := createTestTool(browser)

		Convey("When Schema() is called", func() {
			schema := tool.Schema()

			Convey("Then a valid schema should be returned", func() {
				So(schema, ShouldNotBeNil)
				So(schema["type"], ShouldEqual, "object")

				props, ok := schema["properties"].(map[string]interface{})
				So(ok, ShouldBeTrue)
				So(props, ShouldNotBeEmpty)

				action, ok := props["action"].(map[string]interface{})
				So(ok, ShouldBeTrue)
				So(action["type"], ShouldEqual, "string")

				required, ok := schema["required"].([]string)
				So(ok, ShouldBeTrue)
				So(required, ShouldContain, "action")
			})
		})
	})
}

func TestToolExecute(t *testing.T) {
	Convey("Given a Rod Browser Tool instance", t, func() {
		browser := testSetup(t)
		tool := createTestTool(browser)

		Convey("When Execute() is called without an action", func() {
			args := map[string]interface{}{}
			_, err := tool.Execute(nil, args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "action parameter is required")
			})
		})

		Convey("When Execute() is called with an invalid action", func() {
			args := map[string]interface{}{
				"action": "invalid-action",
			}
			_, err := tool.Execute(nil, args)

			Convey("Then an error should be returned", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "unknown action")
			})
		})
	})
}

func TestToolClose(t *testing.T) {
	// This test creates its own browser to test closing
	Convey("Given a Rod Browser Tool instance with a real browser", t, func() {
		if testing.Short() {
			t.Skip("Skipping test in short mode")
		}

		// Create a new browser instance specifically for this test
		browser := rod.New().Timeout(5 * time.Second).MustConnect()
		tool := &Tool{
			browser: browser,
		}

		Convey("When Close() is called", func() {
			err := tool.Close()

			Convey("Then no error should be returned", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}
