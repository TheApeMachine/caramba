package errnie

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewElasticPostWriter(t *testing.T) {
	Convey("Given newElasticPostWriter", t, func() {
		Convey("When url or index is empty", func() {
			_, err := newElasticPostWriter("", "idx", "", "")
			So(err, ShouldNotBeNil)

			_, err = newElasticPostWriter("http://localhost:9200", "", "", "")
			So(err, ShouldNotBeNil)
		})

		Convey("When url and index are set", func() {
			remote, err := newElasticPostWriter("http://localhost:9200/", "caramba", "u", "p")
			So(err, ShouldBeNil)
			So(remote, ShouldNotBeNil)
			So(remote.endpoint, ShouldEqual, "http://localhost:9200/caramba/_doc")
		})
	})
}

func TestElasticPostWriterWrite(t *testing.T) {
	Convey("Given an Elasticsearch io.Writer sink", t, func() {
		var sawAuth string
		var gotPath string
		var gotBody string

		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
			body, _ := io.ReadAll(request.Body)
			sawAuth = request.Header.Get("Authorization")
			gotPath = request.URL.Path
			gotBody = string(body)
			response.WriteHeader(http.StatusCreated)
		}))
		defer server.Close()

		remote, err := newElasticPostWriter(server.URL, "caramba", "user", "secret")
		So(err, ShouldBeNil)

		n, err := remote.Write([]byte(`{"msg":"hello"}`))
		So(err, ShouldBeNil)
		So(n, ShouldEqual, len(`{"msg":"hello"}`))
		So(sawAuth, ShouldNotBeEmpty)
		So(gotPath, ShouldEqual, "/caramba/_doc")
		So(gotBody, ShouldEqual, `{"msg":"hello"}`)
	})
}
