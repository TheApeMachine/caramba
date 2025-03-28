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
	ArtifactRoleInspect
	ArtifactRoleMetadata
	ArtifactRoleRegistration
	ArtifactRoleSignal
)

type ArtifactScope uint

const (
	ArtifactScopeUnknown ArtifactScope = iota
	ArtifactScopeEvent
	ArtifactScopeMessage
	ArtifactScopePrompt
	ArtifactScopeProvider
	ArtifactScopeName
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

func (artifact *Artifact) Is(role ArtifactRole, scope ArtifactScope) (ok bool) {
	return uint32(role) == artifact.Role() && uint32(scope) == artifact.Scope()
}
