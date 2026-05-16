package config

var hubRootKey = "hub"

type HubConfig struct {
	Endpoint   string
	CacheDir   string
	Token      string
	Offline    bool
	MaxWorkers int
	Xet        HubXetConfig
}

type HubXetConfig struct {
	Active bool
}

/*
NewHubConfig reads Hugging Face Hub settings from the loaded config.yml.
*/
func NewHubConfig() *HubConfig {
	return &HubConfig{
		Endpoint: WithDefault(
			hubRootKey+".endpoint",
			"https://huggingface.co",
		),
		CacheDir: WithDefault(
			hubRootKey+".cache_dir",
			"${HOME}/.cache/huggingface/hub",
		),
		Token: WithDefault(hubRootKey+".token", ""),
		Offline: WithDefault(
			hubRootKey+".offline",
			false,
		),
		MaxWorkers: WithDefault(
			hubRootKey+".max_workers",
			8,
		),
		Xet: HubXetConfig{
			Active: WithDefault(
				hubRootKey+".xet.active",
				true,
			),
		},
	}
}
