package config

var devteamRootKey = "devteam"

/*
ProviderConfig holds the connection details for a single LLM provider.

Provider values: "anthropic", "openai", or any OpenAI-compatible alias
(e.g. "ollama", "groq", "together"). Anything except "anthropic" is routed
through the OpenAI-compatible client.

Set BaseURL to override the default API endpoint — e.g. "http://localhost:11434/v1"
for Ollama, or any vLLM / LM Studio / Together / Groq base URL.
*/
type ProviderConfig struct {
	Provider string
	APIKey   string
	BaseURL  string
	Model    string
}

/*
DevTeamConfig holds settings for the AI development team pipeline.

When Active is false the orchestrator does not start, leaving the kanban board
in a purely human-operated mode.

Developer and Reviewer each get their own ProviderConfig so they can use
different providers, models, or even local endpoints independently.
*/
type DevTeamConfig struct {
	Active            bool
	DatabaseURL       string
	GitHubToken       string
	GitHubOwner       string
	GitHubRepo        string
	RequestsProjectID string
	DockerImage       string
	MaxConcurrent     int
	BlastRadiusDepth  int
	Planner           ProviderConfig
	Developer         ProviderConfig
	Reviewer          ProviderConfig
}

func providerConfig(prefix string) ProviderConfig {
	return ProviderConfig{
		Provider: WithDefault(prefix+".provider", "anthropic"),
		APIKey:   WithDefault(prefix+".api_key", ""),
		BaseURL:  WithDefault(prefix+".base_url", ""),
		Model:    WithDefault(prefix+".model", ""),
	}
}

/*
NewDevTeamConfig reads devteam settings from viper-loaded config.yml.
*/
func NewDevTeamConfig() *DevTeamConfig {
	root := devteamRootKey

	return &DevTeamConfig{
		Active:            WithDefault(root+".active", false),
		DatabaseURL:       WithDefault(root+".database_url", ""),
		GitHubToken:       WithDefault(root+".github_token", ""),
		GitHubOwner:       WithDefault(root+".github_owner", ""),
		GitHubRepo:        WithDefault(root+".github_repo", ""),
		RequestsProjectID: WithDefault(root+".requests_project_id", ""),
		DockerImage:       WithDefault(root+".docker_image", "golang:1.26.1-bookworm"),
		MaxConcurrent:     WithDefault(root+".max_concurrent", 3),
		BlastRadiusDepth:  WithDefault(root+".blast_radius_depth", 2),
		Planner:           providerConfig(root + ".planner"),
		Developer:         providerConfig(root + ".developer"),
		Reviewer:          providerConfig(root + ".reviewer"),
	}
}
