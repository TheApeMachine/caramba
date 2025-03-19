package datura

type ArtifactRole uint

const (
	ArtifactRoleUnknown ArtifactRole = iota
	ArtifactRoleSystem
	ArtifactRoleUser
	ArtifactRoleAssistant
	ArtifactRoleOpenFile
	ArtifactRoleSaveFile
	ArtifactRoleDeleteFile
	ArtifactRoleListFiles
	ArtifactRoleMemoryTool
	ArtifactRoleAgentTool
	ArtifactRoleEditorTool
	ArtifactRoleGithubTool
	ArtifactRoleAzureTool
	ArtifactRoleTrengoTool
	ArtifactRoleBrowserTool
	ArtifactRoleEnvironmentTool
)

type ArtifactScope uint

const (
	ArtifactScopeUnknown ArtifactScope = iota
	ArtifactScopeEvent
	ArtifactScopeMessage
	ArtifactScopePrompt
)

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
