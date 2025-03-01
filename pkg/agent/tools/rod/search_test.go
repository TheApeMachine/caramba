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
var searchTestSetup = func() func(t *testing.T) *rod.Browser {
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

func TestSearch(t *testing.T) {
	Convey("Given a Tool instance", t, func() {
		// Test parameter validation first (doesn't require an actual browser)
		Convey("Parameter validation tests", func() {
			tool := &Tool{}

			Convey("When search is called without a query", func() {
				args := map[string]interface{}{}
				_, err := tool.search(context.Background(), args)

				Convey("Then an error should be returned", func() {
					So(err, ShouldNotBeNil)
					So(err.Error(), ShouldContainSubstring, "query must be a non-empty string")
				})
			})

			Convey("When search is called with an empty query", func() {
				args := map[string]interface{}{
					"query": "",
				}
				_, err := tool.search(context.Background(), args)

				Convey("Then an error should be returned", func() {
					So(err, ShouldNotBeNil)
					So(err.Error(), ShouldContainSubstring, "query must be a non-empty string")
				})
			})

			Convey("When search is called with a non-string query", func() {
				args := map[string]interface{}{
					"query": 123,
				}
				_, err := tool.search(context.Background(), args)

				Convey("Then an error should be returned", func() {
					So(err, ShouldNotBeNil)
					So(err.Error(), ShouldContainSubstring, "query must be a non-empty string")
				})
			})
		})

		// Skip integration tests if running in short mode
		if testing.Short() {
			t.Skip("Skipping integration tests in short mode")
		}

		// Real browser tests
		Convey("Integration tests with real browser", func() {
			browser := searchTestSetup(t)
			tool := &Tool{
				browser: browser,
				timeout: 5 * time.Second,
			}

			Convey("When search is called with a valid query", func() {
				args := map[string]interface{}{
					"query": "example domain",
				}
				resultInterface, err := tool.search(context.Background(), args)

				Convey("Then no error should be returned", func() {
					So(err, ShouldBeNil)
					So(resultInterface, ShouldNotBeNil)

					// Type assertion for the result map
					result, ok := resultInterface.(map[string]interface{})
					So(ok, ShouldBeTrue)

					// Check that result contains search results
					searchResults, ok := result["results"]
					So(ok, ShouldBeTrue)
					So(searchResults, ShouldNotBeNil)
				})
			})
		})
	})
}

func TestExtractSearchResults(t *testing.T) {
	Convey("Given a Tool instance", t, func() {
		tool := &Tool{}

		Convey("When extractSearchResults is called with empty HTML", func() {
			results := tool.extractSearchResults("")

			Convey("Then an empty slice should be returned", func() {
				So(results, ShouldNotBeNil)
				So(len(results), ShouldEqual, 0)
			})
		})

		Convey("When extractSearchResults is called with HTML containing no results", func() {
			html := "<html><body>No results found</body></html>"
			results := tool.extractSearchResults(html)

			Convey("Then an empty slice should be returned", func() {
				So(results, ShouldNotBeNil)
				So(len(results), ShouldEqual, 0)
			})
		})

		Convey("When extractSearchResults is called with HTML containing results", func() {
			html := `<html><body>
				<div class="result">
					<a class="result__a" href="https://example.com">Example Title</a>
					<div class="result__snippet">Example snippet text</div>
				</div><!--result-->
			</body></html>`
			results := tool.extractSearchResults(html)

			Convey("Then results should be extracted", func() {
				So(results, ShouldNotBeNil)
				So(len(results), ShouldEqual, 1)
				So(results[0]["title"], ShouldEqual, "Example Title")
				So(results[0]["url"], ShouldEqual, "https://example.com")
				So(results[0]["snippet"], ShouldEqual, "Example snippet text")
			})
		})

		Convey("When extractSearchResults is called with HTML containing multiple results", func() {
			html := `<html><body>
				<div class="result">
					<a class="result__a" href="https://example1.com">Example Title 1</a>
					<div class="result__snippet">Example snippet text 1</div>
				</div><!--result-->
				<div class="result">
					<a class="result__a" href="https://example2.com">Example Title 2</a>
					<div class="result__snippet">Example snippet text 2</div>
				</div><!--result-->
			</body></html>`
			results := tool.extractSearchResults(html)

			Convey("Then all results should be extracted", func() {
				So(results, ShouldNotBeNil)
				So(len(results), ShouldEqual, 2)
				So(results[0]["title"], ShouldEqual, "Example Title 1")
				So(results[1]["title"], ShouldEqual, "Example Title 2")
			})
		})

		Convey("When extractSearchResults is called with alternative HTML structure", func() {
			html := `<html><body>
				<div class="links_main">
					<a class="result__a" href="https://example.com">Example Title</a>
					<a class="result__snippet">Example snippet text</a>
				</div>
			</body></html>`
			results := tool.extractSearchResults(html)

			Convey("Then results should be extracted using the alternative structure", func() {
				So(results, ShouldNotBeNil)
				So(len(results), ShouldEqual, 1)
			})
		})
	})
}
