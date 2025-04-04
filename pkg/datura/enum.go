package datura

type ArtifactRole uint

const (
	ArtifactRoleUnknown ArtifactRole = iota
	ArtifactRoleSystem
	ArtifactRoleUser
	ArtifactRoleAssistant
	ArtifactRoleTool
	ArtifactRoleQuestion
	ArtifactRoleAnswer
	ArtifactRoleAcknowledge
	ArtifactRoleOpenFile
	ArtifactRoleSaveFile
	ArtifactRoleDeleteFile
	ArtifactRoleListFiles
	ArtifactRoleResponseFormat
)

// ArtifactRoleString returns the string representation of the ArtifactRole.
func (role ArtifactRole) String() string {
	return []string{
		"unknown",
		"system",
		"user",
		"assistant",
		"tool",
		"question",
		"answer",
		"acknowledge",
		"open_file",
		"save_file",
		"delete_file",
		"list_files",
		"response_format",
	}[role]
}

type ArtifactScope uint

const (
	ArtifactScopeUnknown ArtifactScope = iota
	ArtifactScopeGeneration
	ArtifactScopeParams
	ArtifactScopeContext
	ArtifactScopeAquire
	ArtifactScopeRelease
	ArtifactScopePreflight
)

// ArtifactScopeString returns the string representation of the ArtifactScope.
func (scope ArtifactScope) String() string {
	return []string{
		"unknown",
		"generation",
		"params",
		"context",
		"aquire",
		"release",
		"preflight",
	}[scope]
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

func (artifact *Artifact) Is(role ArtifactRole, scope ArtifactScope) (ok bool) {
	return uint32(role) == artifact.Role() && uint32(scope) == artifact.Scope()
}
