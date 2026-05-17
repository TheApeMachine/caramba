package devteam

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

type captureProvider struct {
	request ChatRequest
}

func (provider *captureProvider) Chat(
	ctx context.Context, request ChatRequest,
) (ChatResponse, error) {
	provider.request = request

	return ChatResponse{}, nil
}

func TestAgentAssetsValidate(test *testing.T) {
	Convey("Given the embedded developer-team agent assets", test, func() {
		Convey("It should load valid developer tools and role prompts", func() {
			err := agentAssets.Validate()

			So(err, ShouldBeNil)
			So(agentAssets.Developer.SystemPrompt, ShouldContainSubstring, "{{blast_context}}")
			So(agentAssets.Reviewer.SystemPrompt, ShouldContainSubstring, "JSON object")
			So(agentAssets.Developer.Tools, ShouldHaveLength, 7)
			So(agentAssets.Developer.Tools[0].Name, ShouldEqual, "search_code")
			So(agentAssets.Developer.Tools[1].Name, ShouldEqual, "view_file")
			So(agentAssets.Developer.Tools[6].Name, ShouldEqual, "done")
		})

		Convey("It should reject duplicate tool names", func() {
			assets := agentAssets
			assets.Developer.Tools = append(
				append([]ToolDefinition{}, agentAssets.Developer.Tools...),
				agentAssets.Developer.Tools[0],
			)

			err := assets.Validate()

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "duplicated")
		})
	})
}

func TestDeveloperSystemPrompt(test *testing.T) {
	Convey("Given a Developer and blast radius context", test, func() {
		developer := &Developer{}

		Convey("It should render the embedded prompt with the context inserted", func() {
			systemPrompt := developer.SystemPrompt("pkg/devteam/agent.go owns the agent loop.")

			So(systemPrompt, ShouldContainSubstring, "senior Go software engineer")
			So(systemPrompt, ShouldContainSubstring, "pkg/devteam/agent.go owns the agent loop.")
			So(systemPrompt, ShouldNotContainSubstring, "{{blast_context}}")
		})
	})
}

func TestDeveloperImplement(test *testing.T) {
	Convey("Given a Developer backed by a capture provider", test, func() {
		provider := &captureProvider{}
		developer := &Developer{
			ctx: context.Background(),
			llm: provider,
		}

		Convey("It should send embedded tools and the rendered system prompt", func() {
			err := developer.Implement(
				"Move agent prompts",
				"Load prompts from embedded YAML.",
				"blast context line",
				"",
			)

			So(err, ShouldBeNil)
			So(provider.request.System, ShouldContainSubstring, "blast context line")
			So(provider.request.Tools, ShouldHaveLength, 7)
			So(provider.request.Tools[0].Name, ShouldEqual, "search_code")
		})
	})
}

func TestReviewerSystemPrompt(test *testing.T) {
	Convey("Given a Reviewer", test, func() {
		reviewer := &Reviewer{}

		Convey("It should return the embedded reviewer prompt", func() {
			systemPrompt := reviewer.SystemPrompt()

			So(systemPrompt, ShouldContainSubstring, "senior Go code reviewer")
			So(systemPrompt, ShouldContainSubstring, "JSON object")
		})
	})
}

func BenchmarkAgentAssetsValidate(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = agentAssets.Validate()
	}
}

func BenchmarkDeveloperSystemPrompt(benchmark *testing.B) {
	developer := &Developer{}

	for benchmark.Loop() {
		_ = developer.SystemPrompt("blast context")
	}
}

func BenchmarkReviewerSystemPrompt(benchmark *testing.B) {
	reviewer := &Reviewer{}

	for benchmark.Loop() {
		_ = reviewer.SystemPrompt()
	}
}
