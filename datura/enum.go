package datura

type ArtifactRole uint

const (
	ArtifactRoleUnknown ArtifactRole = iota
	ArtifactRoleSystem
	ArtifactRoleUser
	ArtifactRoleAssistant
	ArtifactRoleMessage
	ArtifactRoleTopic
	ArtifactRoleBroadcast
)

type ArtifactScope uint

const (
	ArtifactScopeUnknown ArtifactScope = iota
	ArtifactScopePrompt
	ArtifactScopeChatCompletion
	ArtifactScopeStructuredOutput
	ArtifactScopeFunctionCall
	ArtifactScopeBroadcast
)

type MediaType string

const (
	MediaTypeUnknown                MediaType = "unknown"
	MediaTypeTextPlain              MediaType = "text/plain"
	MediaTypeApplicationJson        MediaType = "application/json"
	MediaTypeApplicationPdf         MediaType = "application/pdf"
	MediaTypeApplicationOctetStream MediaType = "application/octet-stream"
	MediaTypeApplicationZip         MediaType = "application/zip"
	MediaTypeApplicationGzip        MediaType = "application/gzip"
	MediaTypeApplicationXZip        MediaType = "application/x-zip-compressed"
	MediaTypeXML                    MediaType = "application/xml"
)
