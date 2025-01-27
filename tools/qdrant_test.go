package tools

import (
	"context"
	"net/url"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/tmc/langchaingo/schema"
)

func TestQdrant(t *testing.T) {
	Convey("Given a new Qdrant instance", t, func() {
		collection := "test_collection"
		dimension := uint64(1536)
		qdrant := NewQdrant(collection, dimension)

		Convey("When creating a new instance", func() {
			Convey("Then it should be properly initialized", func() {
				So(qdrant, ShouldNotBeNil)
				So(qdrant.ctx, ShouldNotBeNil)
				So(qdrant.client, ShouldNotBeNil)
				So(qdrant.embedder, ShouldNotBeNil)
				So(qdrant.collection, ShouldEqual, collection)
				So(qdrant.dimension, ShouldEqual, dimension)
			})
		})

		Convey("When initializing", func() {
			err := qdrant.Initialize()

			Convey("Then it should initialize successfully", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When connecting", func() {
			err := qdrant.Connect()

			Convey("Then it should connect successfully", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When using the tool", func() {
			Convey("With add operation", func() {
				args := map[string]any{
					"operation": "add",
					"documents": []string{"test document"},
					"metadata": map[string]any{
						"test": "metadata",
					},
				}
				result := qdrant.Use(context.Background(), args)

				Convey("Then it should handle document addition", func() {
					So(result, ShouldEqual, "memory saved in vector store")
				})
			})

			Convey("With query operation", func() {
				args := map[string]any{
					"operation": "query",
					"query":     "test query",
				}
				result := qdrant.Use(context.Background(), args)

				Convey("Then it should handle query operation", func() {
					So(result, ShouldNotBeEmpty)
				})
			})

			Convey("With invalid operation", func() {
				args := map[string]any{
					"operation": "invalid",
				}
				result := qdrant.Use(context.Background(), args)

				Convey("Then it should return error message", func() {
					So(result, ShouldEqual, "Unsupported operation")
				})
			})
		})

		Convey("When adding documents", func() {
			docs := []schema.Document{
				{
					PageContent: "test content",
					Metadata: map[string]any{
						"test": "metadata",
					},
				},
			}
			err := qdrant.AddDocuments(docs)

			Convey("Then it should add documents successfully", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When performing similarity search", func() {
			query := "test query"
			k := 1
			docs, err := qdrant.SimilaritySearch(query, k)

			Convey("Then it should perform search successfully", func() {
				So(err, ShouldBeNil)
				So(docs, ShouldNotBeNil)
			})
		})

		Convey("When querying", func() {
			query := "test query"
			results, err := qdrant.Query(query)

			Convey("Then it should perform query successfully", func() {
				So(err, ShouldBeNil)
				So(results, ShouldNotBeNil)
			})
		})

		Convey("When adding content", func() {
			docs := []string{"test document"}
			metadata := map[string]any{"test": "metadata"}
			result := qdrant.Add(docs, metadata)

			Convey("Then it should add content successfully", func() {
				So(result, ShouldEqual, "memory saved in vector store")
			})
		})

		Convey("When creating collection if not exists", func() {
			uri, _ := url.Parse("http://localhost:6333")
			err := createCollectionIfNotExists(collection, uri, dimension)

			Convey("Then it should handle collection creation", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}
