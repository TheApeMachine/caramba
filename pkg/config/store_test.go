package config

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestNewQdrantStoreConfig(test *testing.T) {
	Convey("Given store.qdrant settings", test, func() {
		setStoreConfigValue(test, "store.qdrant.url", "http://example:6333")
		setStoreConfigValue(test, "store.qdrant.api_key", "qkey")
		setStoreConfigValue(test, "store.qdrant.use_tls", true)
		setStoreConfigValue(test, "store.qdrant.grpc_port", 7777)

		qdrantConfig := NewQdrantStoreConfig()

		Convey("It should load Qdrant settings from config", func() {
			So(qdrantConfig.URL, ShouldEqual, "http://example:6333")
			So(qdrantConfig.APIKey, ShouldEqual, "qkey")
			So(qdrantConfig.UseTLS, ShouldBeTrue)
			So(qdrantConfig.GRPCPort, ShouldEqual, 7777)
		})
	})
}

func TestNewNeo4jStoreConfig(test *testing.T) {
	Convey("Given store.neo4j settings", test, func() {
		setStoreConfigValue(test, "store.neo4j.uri", "neo4j://localhost:7687")
		setStoreConfigValue(test, "store.neo4j.username", "neo4j")
		setStoreConfigValue(test, "store.neo4j.password", "secret")
		setStoreConfigValue(test, "store.neo4j.database", "graph")

		neo4jConfig := NewNeo4jStoreConfig()

		Convey("It should load Neo4j settings from config", func() {
			So(neo4jConfig.URI, ShouldEqual, "neo4j://localhost:7687")
			So(neo4jConfig.Username, ShouldEqual, "neo4j")
			So(neo4jConfig.Password, ShouldEqual, "secret")
			So(neo4jConfig.Database, ShouldEqual, "graph")
		})
	})
}

func TestNewElasticsearchStoreConfig(test *testing.T) {
	Convey("Given store.elasticsearch settings", test, func() {
		setStoreConfigValue(test, "store.elasticsearch.url", "http://localhost:9200")
		setStoreConfigValue(test, "store.elasticsearch.username", "elastic")
		setStoreConfigValue(test, "store.elasticsearch.api_key", "es-key")

		elasticsearchConfig := NewElasticsearchStoreConfig()

		Convey("It should load Elasticsearch settings from config", func() {
			So(elasticsearchConfig.URL, ShouldEqual, "http://localhost:9200")
			So(elasticsearchConfig.Username, ShouldEqual, "elastic")
			So(elasticsearchConfig.APIKey, ShouldEqual, "es-key")
		})
	})
}

func TestNewDeeplakeStoreConfig(test *testing.T) {
	Convey("Given store.deeplake settings", test, func() {
		setStoreConfigValue(test, "store.deeplake.api_url", "https://api.deeplake.ai")
		setStoreConfigValue(test, "store.deeplake.api_key", "dl-key")
		setStoreConfigValue(test, "store.deeplake.org_id", "org-1")
		setStoreConfigValue(test, "store.deeplake.workspace", "default")

		deeplakeConfig := NewDeeplakeStoreConfig()

		Convey("It should load DeepLake settings from config", func() {
			So(deeplakeConfig.APIURL, ShouldEqual, "https://api.deeplake.ai")
			So(deeplakeConfig.APIKey, ShouldEqual, "dl-key")
			So(deeplakeConfig.OrgID, ShouldEqual, "org-1")
			So(deeplakeConfig.Workspace, ShouldEqual, "default")
		})
	})
}

func BenchmarkNewStoreConfig(benchmark *testing.B) {
	setStoreConfigValue(benchmark, "store.qdrant.host", "localhost")

	for benchmark.Loop() {
		_ = NewStoreConfig()
	}
}

func setStoreConfigValue(testingHandle interface {
	Helper()
	Cleanup(func())
}, key string, value any) {
	testingHandle.Helper()

	viper.Set(key, value)
	testingHandle.Cleanup(func() {
		viper.Set(key, nil)
	})
}
