package rod

import (
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func TestGetTimeoutDuration(t *testing.T) {
	Convey("Given a map of arguments", t, func() {
		defaultTimeout := 30 * time.Second

		Convey("When an integer timeout is provided", func() {
			args := map[string]interface{}{
				"timeout": 10,
			}
			timeout := getTimeoutDuration(args, defaultTimeout)

			Convey("Then the correct duration should be returned", func() {
				So(timeout, ShouldEqual, 10*time.Second)
			})
		})

		Convey("When a float timeout is provided", func() {
			args := map[string]interface{}{
				"timeout": 5.5,
			}
			timeout := getTimeoutDuration(args, defaultTimeout)

			Convey("Then the correct duration should be returned", func() {
				So(timeout, ShouldEqual, 5*time.Second+500*time.Millisecond)
			})
		})

		Convey("When no timeout is provided", func() {
			args := map[string]interface{}{}
			timeout := getTimeoutDuration(args, defaultTimeout)

			Convey("Then the default duration should be returned", func() {
				So(timeout, ShouldEqual, defaultTimeout)
			})
		})
	})
}

func TestGetStringOr(t *testing.T) {
	Convey("Given a map of arguments", t, func() {
		args := map[string]interface{}{
			"validKey": "validValue",
			"emptyKey": "",
		}

		Convey("When a valid key with non-empty value is provided", func() {
			value := getStringOr(args, "validKey", "default")

			Convey("Then the actual value should be returned", func() {
				So(value, ShouldEqual, "validValue")
			})
		})

		Convey("When a key with empty value is provided", func() {
			value := getStringOr(args, "emptyKey", "default")

			Convey("Then the default value should be returned", func() {
				So(value, ShouldEqual, "default")
			})
		})

		Convey("When a non-existent key is provided", func() {
			value := getStringOr(args, "nonExistentKey", "default")

			Convey("Then the default value should be returned", func() {
				So(value, ShouldEqual, "default")
			})
		})
	})
}

func TestGetBoolOr(t *testing.T) {
	Convey("Given a map of arguments", t, func() {
		args := map[string]interface{}{
			"trueKey":  true,
			"falseKey": false,
		}

		Convey("When a key with true value is provided", func() {
			value := getBoolOr(args, "trueKey", false)

			Convey("Then true should be returned", func() {
				So(value, ShouldBeTrue)
			})
		})

		Convey("When a key with false value is provided", func() {
			value := getBoolOr(args, "falseKey", true)

			Convey("Then false should be returned", func() {
				So(value, ShouldBeFalse)
			})
		})

		Convey("When a non-existent key is provided", func() {
			value := getBoolOr(args, "nonExistentKey", true)

			Convey("Then the default value should be returned", func() {
				So(value, ShouldBeTrue)
			})
		})
	})
}

func TestExtractBetween(t *testing.T) {
	Convey("Given a text and boundary strings", t, func() {
		text := "This is a <tag>value</tag> in a text"

		Convey("When extracting content between existing boundaries", func() {
			extracted := extractBetween(text, "<tag>", "</tag>")

			Convey("Then the correct content should be returned", func() {
				So(extracted, ShouldEqual, "value")
			})
		})

		Convey("When start boundary doesn't exist", func() {
			extracted := extractBetween(text, "<missing>", "</tag>")

			Convey("Then an empty string should be returned", func() {
				So(extracted, ShouldEqual, "")
			})
		})

		Convey("When end boundary doesn't exist", func() {
			extracted := extractBetween(text, "<tag>", "</missing>")

			Convey("Then an empty string should be returned", func() {
				So(extracted, ShouldEqual, "")
			})
		})
	})
}

func TestExtractBetweenAll(t *testing.T) {
	Convey("Given a text with multiple occurrences of boundary strings", t, func() {
		text := "<div>first</div><div>second</div><div>third</div>"

		Convey("When extracting all content between existing boundaries", func() {
			extracted := extractBetweenAll(text, "<div>", "</div>")

			Convey("Then all matching contents should be returned", func() {
				So(len(extracted), ShouldEqual, 3)
				So(extracted[0], ShouldEqual, "first")
				So(extracted[1], ShouldEqual, "second")
				So(extracted[2], ShouldEqual, "third")
			})
		})

		Convey("When boundaries don't exist", func() {
			extracted := extractBetweenAll(text, "<missing>", "</missing>")

			Convey("Then an empty slice should be returned", func() {
				So(len(extracted), ShouldEqual, 0)
			})
		})
	})
}

func TestStripTags(t *testing.T) {
	Convey("Given a text with HTML tags", t, func() {
		Convey("When stripping simple tags", func() {
			text := "<div>content</div>"
			stripped := stripTags(text)

			Convey("Then tags should be removed", func() {
				So(stripped, ShouldEqual, "content")
			})
		})

		Convey("When stripping nested tags", func() {
			text := "<div>outer<span>inner</span>content</div>"
			stripped := stripTags(text)

			Convey("Then all tags should be removed", func() {
				So(stripped, ShouldEqual, "outerinnercontent")
			})
		})

		Convey("When stripping text with attributes", func() {
			text := "<div class=\"test\">content</div>"
			stripped := stripTags(text)

			Convey("Then tags with attributes should be removed", func() {
				So(stripped, ShouldEqual, "content")
			})
		})
	})
}
