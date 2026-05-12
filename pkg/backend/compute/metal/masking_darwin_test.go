//go:build darwin && cgo

package metal

import (
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMetalMasking_ForwardErrors(t *testing.T) {
	Convey("Given Metal masking cannot initialise", t, func() {
		masking, err := NewMasking(filepath.Join(t.TempDir(), "missing.metallib"))

		Convey("It should fail before any scalar fallback can run", func() {
			So(err, ShouldNotBeNil)
			So(masking, ShouldBeNil)
		})
	})
}
