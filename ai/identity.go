package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/minio/minio-go/v7"
	"github.com/spf13/viper"
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
	Params *provider.GenerationParams
	conn   *datalake.Conn
	ctx    context.Context
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
*/
func NewIdentity(ctx context.Context, role string) *Identity {
	return &Identity{
		Role: role,
		conn: datalake.NewConn(),
		ctx:  ctx,
	}
}

/*
Initialize loads an existing identity from storage or creates a new one if none exists.
It ensures the Identity has valid generation parameters, using defaults if necessary.
Returns the initialized Identity instance.
*/
func (identity *Identity) Initialize() *Identity {
	if ok := identity.load(); !ok {
		identity.create()
	}

	if identity.Params == nil {
		errnie.Warn("identity has no params, going with defaults")
		identity.Params = provider.NewGenerationParams()
	}

	return identity
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
	if identity.loaded, identity.err = identity.conn.Get(identity.ctx, "identities/"+identity.Role); identity.err != nil {
		errnie.Info("no existing identity found for %s", identity.Role)
		return false
	}

	identity.Params = &provider.GenerationParams{}
	json.NewDecoder(identity.loaded).Decode(identity.Params)

	errnie.Info("identity loaded %s (%s)", identity.Name, identity.Role)
	return true
}

/*
create generates a new Identity with system prompts from configuration,
a new unique name, and default generation parameters.
It validates and saves the new identity to storage.
*/
func (identity *Identity) create() {
	v := viper.GetViper()

	subkey := "unstructured"

	if identity.Params.Process != nil {
		subkey = "structured"
	}

	identity.System = v.GetString("prompts.system." + subkey)
	identity.Name = utils.NewName()
	identity.Params = provider.NewGenerationParams()

	if identity.err = identity.validate(); identity.err != nil {
		errnie.Error(identity.err)
		return
	}

	identity.save()
	errnie.Info("identity created %s (%s)", identity.Name, identity.Role)
}

/*
save persists the current Identity's parameters to storage,
using the role as the storage key.
*/
func (identity *Identity) save() {
	var buf bytes.Buffer
	json.NewEncoder(&buf).Encode(identity.Params)

	identity.conn.Put(identity.ctx, "identities/"+identity.Role, buf.Bytes(), nil)
	errnie.Info("identity saved %s (%s)", identity.Name, identity.Role)
}

/*
validate checks if all required fields of the Identity are properly set.
Returns an error if any required field is missing or invalid.
*/
func (identity *Identity) validate() error {
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
