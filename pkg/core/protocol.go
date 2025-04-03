package core

import (
	"strings"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type Protocol struct {
	ID          string
	sender      string
	receiver    string
	raw         []string
	steps       []Step
	currentStep int
}

type Step struct {
	From   string
	To     string
	Role   datura.ArtifactRole
	Scope  datura.ArtifactScope
	Status Status
}

func NewProtocol(key string, sender string, receiver string) *Protocol {
	errnie.Debug("core.NewProtocol", "key", key)

	return &Protocol{
		ID:          uuid.New().String(),
		raw:         tweaker.GetProtocol(key),
		steps:       make([]Step, 0),
		currentStep: -1,
		sender:      sender,
		receiver:    receiver,
	}
}

func (protocol *Protocol) HandleMessage(
	from string,
	artifact *datura.Artifact,
) (*datura.Artifact, Status) {
	errnie.Debug("core.Protocol.HandleMessage", "from", from)

	if len(protocol.steps) == 0 {
		protocol.parse()
	}

	protocol.currentStep++
	step := protocol.steps[protocol.currentStep]

	errnie.Info(
		"core.Protocol.HandleMessage",
		"role", step.Role,
		"scope", step.Scope,
		"from", step.From,
		"to", step.To,
		"status", step.Status,
	)

	return datura.New(
		datura.WithRole(step.Role),
		datura.WithScope(step.Scope),
		datura.WithMeta("from", step.From),
		datura.WithMeta("to", step.To),
		datura.WithMeta("protocol", protocol.ID),
	), step.Status
}

func (protocol *Protocol) parse() {
	errnie.Debug("core.Protocol.parse")

	for _, step := range protocol.raw {
		step = strings.ReplaceAll(step, "sender", protocol.sender)
		step = strings.ReplaceAll(step, "receiver", protocol.receiver)

		chunks := strings.Split(step, " ")
		from := chunks[0]
		to := chunks[2]
		role := protocol.ConvertRole(chunks[1])
		scope := protocol.ConvertScope(chunks[1])
		status := protocol.ConvertStatus(chunks[3])

		from = strings.ReplaceAll(from, "<", "")
		from = strings.ReplaceAll(from, ">", "")
		to = strings.ReplaceAll(to, "<", "")
		to = strings.ReplaceAll(to, ">", "")

		protocol.steps = append(protocol.steps, Step{
			From:   from,
			To:     to,
			Role:   role,
			Scope:  scope,
			Status: status,
		})
	}
}

func (protocol *Protocol) ConvertRole(role string) datura.ArtifactRole {
	role = strings.ReplaceAll(role, "-[", "")
	role = strings.ReplaceAll(role, "]->", "")
	role = strings.Split(role, "|")[0]

	switch role {
	case "question":
		return datura.ArtifactRoleQuestion
	case "answer":
		return datura.ArtifactRoleAnswer
	case "acknowledge":
		return datura.ArtifactRoleAcknowledge
	case "tool":
		return datura.ArtifactRoleTool
	case "open_file":
		return datura.ArtifactRoleOpenFile
	case "save_file":
		return datura.ArtifactRoleSaveFile
	case "delete_file":
		return datura.ArtifactRoleDeleteFile
	case "list_files":
		return datura.ArtifactRoleListFiles
	case "response_format":
		return datura.ArtifactRoleResponseFormat
	case "system":
		return datura.ArtifactRoleSystem
	case "user":
		return datura.ArtifactRoleUser
	case "assistant":
		return datura.ArtifactRoleAssistant
	}

	return datura.ArtifactRoleUnknown
}

func (protocol *Protocol) ConvertScope(scope string) datura.ArtifactScope {
	scope = strings.ReplaceAll(scope, "-[", "")
	scope = strings.ReplaceAll(scope, "]->", "")
	scope = strings.Split(scope, "|")[1]

	switch scope {
	case "context":
		return datura.ArtifactScopeContext
	case "generation":
		return datura.ArtifactScopeGeneration
	case "aquire":
		return datura.ArtifactScopeAquire
	case "release":
		return datura.ArtifactScopeRelease
	case "preflight":
		return datura.ArtifactScopePreflight
	case "params":
		return datura.ArtifactScopeParams
	}

	return datura.ArtifactScopeUnknown
}

func (protocol *Protocol) ConvertStatus(status string) Status {
	status = strings.ReplaceAll(status, "(", "")
	status = strings.ReplaceAll(status, ")", "")

	switch status {
	case "waiting":
		return StatusWaiting
	case "busy":
		return StatusBusy
	case "ready":
		return StatusReady
	case "working":
		return StatusWorking
	case "done":
		return StatusDone
	}

	return StatusUnknown
}

func (protocol *Protocol) GetStatus() Status {
	if len(protocol.steps) == 0 {
		return StatusUnknown
	}
	return protocol.steps[protocol.currentStep].Status
}
