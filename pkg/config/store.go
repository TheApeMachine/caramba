package config

var storeRootKey = "store"

/*
StoreConfig groups connection settings for optional data-store clients.
*/
type StoreConfig struct {
	Qdrant        QdrantStoreConfig
	Neo4j         Neo4jStoreConfig
	Elasticsearch ElasticsearchStoreConfig
	Deeplake      DeeplakeStoreConfig
}

/*
QdrantStoreConfig holds gRPC client settings for Qdrant.
*/
type QdrantStoreConfig struct {
	URL      string
	BaseURL  string
	Host     string
	GRPCPort int
	Port     int
	APIKey   string
	UseTLS   bool
	PoolSize int
}

/*
Neo4jStoreConfig holds Bolt driver settings for Neo4j.
*/
type Neo4jStoreConfig struct {
	URI      string
	Username string
	Password string
	Database string
}

/*
ElasticsearchStoreConfig holds cluster connection settings.

Addresses is a comma-separated list of node URLs. When empty, URL is used as a
single address.
*/
type ElasticsearchStoreConfig struct {
	Addresses string
	URL       string
	Username  string
	Password  string
	APIKey    string
}

/*
DeeplakeStoreConfig holds DeepLake HTTP API settings.
*/
type DeeplakeStoreConfig struct {
	APIURL    string
	APIKey    string
	OrgID     string
	Workspace string
}

/*
NewStoreConfig reads all store.* settings from viper-loaded config.yml.
*/
func NewStoreConfig() *StoreConfig {
	return &StoreConfig{
		Qdrant:        NewQdrantStoreConfig(),
		Neo4j:         NewNeo4jStoreConfig(),
		Elasticsearch: NewElasticsearchStoreConfig(),
		Deeplake:      NewDeeplakeStoreConfig(),
	}
}

/*
NewQdrantStoreConfig reads store.qdrant.* from config.yml.
*/
func NewQdrantStoreConfig() QdrantStoreConfig {
	root := storeRootKey + ".qdrant"

	return QdrantStoreConfig{
		URL:      WithDefault(root+".url", ""),
		BaseURL:  WithDefault(root+".base_url", ""),
		Host:     WithDefault(root+".host", ""),
		GRPCPort: WithDefault(root+".grpc_port", 0),
		Port:     WithDefault(root+".port", 0),
		APIKey:   WithDefault(root+".api_key", ""),
		UseTLS:   WithDefault(root+".use_tls", false),
		PoolSize: WithDefault(root+".pool_size", 0),
	}
}

/*
NewNeo4jStoreConfig reads store.neo4j.* from config.yml.
*/
func NewNeo4jStoreConfig() Neo4jStoreConfig {
	root := storeRootKey + ".neo4j"

	return Neo4jStoreConfig{
		URI:      WithDefault(root+".uri", ""),
		Username: WithDefault(root+".username", ""),
		Password: WithDefault(root+".password", ""),
		Database: WithDefault(root+".database", ""),
	}
}

/*
NewElasticsearchStoreConfig reads store.elasticsearch.* from config.yml.
*/
func NewElasticsearchStoreConfig() ElasticsearchStoreConfig {
	root := storeRootKey + ".elasticsearch"

	return ElasticsearchStoreConfig{
		Addresses: WithDefault(root+".addresses", ""),
		URL:       WithDefault(root+".url", ""),
		Username:  WithDefault(root+".username", ""),
		Password:  WithDefault(root+".password", ""),
		APIKey:    WithDefault(root+".api_key", ""),
	}
}

/*
NewDeeplakeStoreConfig reads store.deeplake.* from config.yml.
*/
func NewDeeplakeStoreConfig() DeeplakeStoreConfig {
	root := storeRootKey + ".deeplake"

	return DeeplakeStoreConfig{
		APIURL:    WithDefault(root+".api_url", ""),
		APIKey:    WithDefault(root+".api_key", ""),
		OrgID:     WithDefault(root+".org_id", ""),
		Workspace: WithDefault(root+".workspace", ""),
	}
}
