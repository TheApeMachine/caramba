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
	ArtifactRoleListRepositories
	ArtifactRoleGetRepository
	ArtifactRoleCreateRepository
	ArtifactRoleListBranches
	ArtifactRoleGetContents
	ArtifactRoleListPullRequests
	ArtifactRoleGetPullRequest
	ArtifactRoleCreatePullRequest
	ArtifactRoleUpdatePullRequest
	ArtifactRoleListIssues
	ArtifactRoleGetIssue
	ArtifactRoleCreateIssue
	ArtifactRoleUpdateIssue
	ArtifactRoleCreatePRComment
	ArtifactRoleListPRComments
	ArtifactRoleCreatePRReview
	ArtifactRoleListPRReviews
	ArtifactRoleCreateReviewComment
	ArtifactRoleListReviewComments
	ArtifactRoleSubmitReview
	ArtifactRoleCreateWorkItem
	ArtifactRoleUpdateWorkItem
	ArtifactRoleGetWorkItem
	ArtifactRoleListWorkItems
	ArtifactRoleCreateWikiPage
	ArtifactRoleUpdateWikiPage
	ArtifactRoleGetWikiPage
	ArtifactRoleListWikiPages
	ArtifactRoleListTickets
	ArtifactRoleCreateTicket
	ArtifactRoleAssignTicket
	ArtifactRoleCloseTicket
	ArtifactRoleReopenTicket
	ArtifactRoleListLabels
	ArtifactRoleGetLabel
	ArtifactRoleCreateLabel
	ArtifactRoleUpdateLabel
	ArtifactRoleDeleteLabel
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
