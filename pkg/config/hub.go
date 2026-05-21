package config

import "github.com/theapemachine/hf/hub"

var hubRootKey = "hub"

/*
NewHubConfig reads Hugging Face Hub settings from the loaded config.yml.
*/
func NewHubConfig() *hub.HubConfig {
	return &hub.HubConfig{
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
		Xet: hub.HubXetConfig{
			Active: WithDefault(
				hubRootKey+".xet.active",
				true,
			),
		},
	}
}
