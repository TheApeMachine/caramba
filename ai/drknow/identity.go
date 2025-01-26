package drknow

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/minio/minio-go/v7"
	"github.com/theapemachine/caramba/datalake"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
Identity provides a trackable range of parameters for an AI agent,
which is needed to enable the self-optimization process.

Each Identity contains configuration for system prompts, naming, role definition,
and generation parameters that control the agent's behavior.
*/
type Identity struct {
	System string `json:"system" jsonschema:"title=System,description=The system prompt for the agent,required"`
	Name   string `json:"name" jsonschema:"title=Name,description=A unique name for the agent,required"`
	Role   string `json:"role" jsonschema:"title=Role,description=The role of the agent,required"`
	Params *provider.LLMGenerationParams
	conn   *datalake.Conn
	Ctx    context.Context
	err    error
	loaded *minio.Object
}

/*
NewIdentity creates a new Identity instance with the specified role and context.
It returns an initialized but not yet loaded Identity that needs to be further
initialized using the Initialize method.

Parameters:
  - ctx: The context for operations
  - role: The role designation for the AI agent
  - system: The system prompt for the AI agent
*/
func NewIdentity(ctx context.Context, role string, system string) *Identity {
	params := provider.NewGenerationParams()
	params.Thread = provider.NewThread(
		provider.NewMessage(
			provider.RoleSystem,
			system,
		),
	)

	return &Identity{
		System: system,
		Name:   utils.NewName(),
		Role:   role,
		Params: params,
		conn:   datalake.NewConn(),
		Ctx:    ctx,
	}
}

/*
String implements the Stringer interface, providing a human-readable
representation of the Identity including its name and role.
*/
func (identity *Identity) String() string {
	return utils.JoinWith(
		"\n",
		fmt.Sprintf("Name: %s", identity.Name),
		fmt.Sprintf("Role: %s", identity.Role),
	)
}

/*
load attempts to retrieve an existing identity from storage based on its role.
Returns true if successful, false if the identity doesn't exist or couldn't be loaded.
*/
func (identity *Identity) load() bool {
	if identity.loaded, identity.err = identity.conn.Get(identity.Ctx, "identities/"+identity.Role); identity.err != nil {
		return false
	}

	if err := json.NewDecoder(identity.loaded).Decode(identity); err != nil {
		errnie.Error(errors.New("failed to decode identity params: " + err.Error()))
		return false
	}

	return true
}

/*
create generates a new Identity with system prompts from configuration,
a new unique name, and default generation parameters.
It validates and saves the new identity to storage.
*/
func (identity *Identity) create() {
	identity.Name = utils.NewName()
	identity.Params = provider.NewGenerationParams()

	if identity.err = identity.Validate(); identity.err != nil {
		errnie.Error(identity.err)
		return
	}

	identity.save()
}

/*
save persists the current Identity's parameters to storage,
using the role as the storage key.
*/
func (identity *Identity) save() {
	if !identity.conn.IsConnected() {
		errnie.Warn("no connection to datalake, skipping save")
		return
	}

	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(identity); err != nil {
		errnie.Error(err)
		return
	}

	if err := identity.conn.Put(identity.Ctx, "identities/"+identity.Role, buf.Bytes(), nil); err != nil {
		errnie.Error(err)
		return
	}
}

/*
Validate checks if all required fields of the Identity are properly set.
Returns an error if any required field is missing or invalid.
*/
func (identity *Identity) Validate() error {
	if identity.System == "" {
		return errors.New("system is required")
	}

	if identity.Name == "" {
		return errors.New("name is required")
	}

	if identity.Role == "" {
		return errors.New("role is required")
	}

	if identity.Params == nil {
		return errors.New("params are required")
	}

	if identity.Params.Thread == nil {
		return errors.New("thread is required")
	}

	return nil
}
