package config

var (
	rootKey          = "errnie"
	fileKey          = rootKey + ".file"
	elasticsearchKey = rootKey + ".elasticsearch"
)

type ErrnieConfig struct {
	Level         string
	File          FileConfig
	Elasticsearch ElasticsearchConfig
}

type FileConfig struct {
	Active bool
	Path   string
}

type ElasticsearchConfig struct {
	Active   bool
	URL      string
	Index    string
	Username string
	Password string
}

func NewErrnieConfig() *ErrnieConfig {
	return &ErrnieConfig{
		Level: WithDefault(rootKey+".level", "info"),
		File: FileConfig{
			Active: WithDefault(fileKey+".active", false),
			Path:   WithDefault(fileKey+".path", ""),
		},
		Elasticsearch: ElasticsearchConfig{
			Active:   WithDefault(elasticsearchKey+".active", false),
			URL:      WithDefault(elasticsearchKey+".url", ""),
			Index:    WithDefault(elasticsearchKey+".index", ""),
			Username: WithDefault(elasticsearchKey+".username", ""),
			Password: WithDefault(elasticsearchKey+".password", ""),
		},
	}
}
