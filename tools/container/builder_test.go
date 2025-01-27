package container

import (
	"context"
	"io"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewBuilder(t *testing.T) {
	Convey("When creating a new Builder", t, func() {
		builder := NewBuilder()

		Convey("Then it should be properly initialized", func() {
			So(builder, ShouldNotBeNil)
			So(builder.client, ShouldNotBeNil)
		})
	})
}

func TestBuildImage(t *testing.T) {
	Convey("Given a Builder instance", t, func() {
		ctx := context.Background()
		builder := NewBuilder()

		Convey("When building an image", func() {
			err := builder.BuildImage(ctx, filepath.Join(".", "Dockerfile"), DefaultImageName)

			Convey("Then it should build successfully", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When building with an invalid path", func() {
			err := builder.BuildImage(ctx, "/nonexistent/path/Dockerfile", DefaultImageName)

			Convey("Then it should return an error", func() {
				So(err, ShouldNotBeNil)
			})
		})
	})
}

func TestProcessAndPrintBuildOutput(t *testing.T) {
	Convey("Given a Builder instance", t, func() {
		builder := NewBuilder()

		Convey("When processing build output", func() {
			mockReader := &mockReader{
				data: []byte(`{"stream":"Step 1/5"}\n{"stream":"Step 2/5"}\n`),
			}

			err := builder.processAndPrintBuildOutput(mockReader)

			Convey("Then it should process without errors", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When processing error output", func() {
			mockReader := &mockReader{
				data: []byte(`{"error":"build failed"}`),
			}

			err := builder.processAndPrintBuildOutput(mockReader)

			Convey("Then it should return the error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "build failed")
			})
		})
	})
}

// mockReader implements io.Reader for testing
type mockReader struct {
	data   []byte
	offset int
}

func (m *mockReader) Read(p []byte) (n int, err error) {
	if m.offset >= len(m.data) {
		return 0, io.EOF
	}
	n = copy(p, m.data[m.offset:])
	m.offset += n
	return n, nil
}
