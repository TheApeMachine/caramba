package tools

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/agent/tools/browser"
)

// TestBrowserToolWrapper tests the wrapper functionality
func TestBrowserToolWrapper(t *testing.T) {
	Convey("Given a BrowserTool with a real browser tool", t, func() {
		// Create a real browser tool with dummy values
		// This is not ideal but avoids the type assertion issues
		browserTool := browser.New("", "")
		tool := &BrowserTool{
			browserTool: browserTool,
		}

		Convey("When Name() is called", func() {
			name := tool.Name()

			Convey("Then it should delegate to the underlying tool", func() {
				So(name, ShouldEqual, browserTool.Name())
			})
		})

		Convey("When Description() is called", func() {
			desc := tool.Description()

			Convey("Then it should delegate to the underlying tool", func() {
				So(desc, ShouldEqual, browserTool.Description())
			})
		})

		Convey("When Schema() is called", func() {
			schema := tool.Schema()

			Convey("Then it should delegate to the underlying tool", func() {
				So(schema, ShouldResemble, browserTool.Schema())
			})
		})

		// Note: We can't easily test Execute() without making actual browser calls
		// In a real-world scenario, we would use dependency injection and interfaces
		// to make this more testable.
	})
}

// TestNewBrowserTool tests the constructor
func TestNewBrowserTool(t *testing.T) {
	Convey("Given browserless URL and API key", t, func() {
		browserlessURL := "https://example.com"
		browserlessAPIKey := "test-api-key"

		Convey("When NewBrowserTool is called", func() {
			tool := NewBrowserTool(browserlessURL, browserlessAPIKey)

			Convey("Then a new BrowserTool should be returned", func() {
				So(tool, ShouldNotBeNil)
				So(tool.browserTool, ShouldNotBeNil)
			})
		})
	})
}
