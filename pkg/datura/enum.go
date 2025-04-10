package datura

type ArtifactRole uint

const (
	ArtifactRoleUnknown ArtifactRole = iota
	ArtifactRoleSystem
	ArtifactRoleUser
	ArtifactRoleAssistant
	ArtifactRoleTool
	ArtifactRoleResource
	ArtifactRolePrompt
	ArtifactRoleSubscriber
	ArtifactRoleAcknowledger
	ArtifactRolePublisher
)

// ArtifactRoleString returns the string representation of the ArtifactRole.
func (role ArtifactRole) String() string {
	return []string{
		"unknown",
		"system",
		"user",
		"assistant",
		"tool",
		"resource",
		"prompt",
		"subscriber",
		"acknowledger",
		"publisher",
	}[role]
}

func (artifact *Artifact) ActsAs(role ArtifactRole) bool {
	return artifact.Role() == uint32(role)
}

type ArtifactScope uint

const (
	ArtifactScopeUnknown ArtifactScope = iota
	ArtifactScopeError
	ArtifactScopeGeneration
	ArtifactScopeParams
	ArtifactScopeContext
	ArtifactScopeAgent
	ArtifactScopeTool
	ArtifactScopeProvider
	ArtifactScopeTopic
	ArtifactScopeResult
)

// ArtifactScopeString returns the string representation of the ArtifactScope.
func (scope ArtifactScope) String() string {
	return []string{
		"unknown",
		"generation",
		"params",
		"context",
		"agent",
		"tool",
		"provider",
		"topic",
		"result",
	}[scope]
}

func (artifact *Artifact) ScopedAs(scope ArtifactScope) bool {
	return artifact.Scope() == uint32(scope)
}

type MediaType string

const (
	MediaTypeUnknown                MediaType = "unknown"
	MediaTypeTextPlain              MediaType = "text/plain"
	MediaTypeApplicationJson        MediaType = "application/json"
	MediaTypeApplicationYaml        MediaType = "application/yaml"
	MediaTypeApplicationXml         MediaType = "application/xml"
	MediaTypeApplicationPdf         MediaType = "application/pdf"
	MediaTypeApplicationOctetStream MediaType = "application/octet-stream"
	MediaTypeCapnp                  MediaType = "application/capnp"
	MediaTypeApplicationZip         MediaType = "application/zip"
	MediaTypeApplicationGzip        MediaType = "application/gzip"
	MediaTypeApplicationXZip        MediaType = "application/x-zip-compressed"
)
