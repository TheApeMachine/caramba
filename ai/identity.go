package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/minio/minio-go/v7"
	"github.com/theapemachine/caramba/datalake"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
Identity provides a trackable range of parameters for an AI agent,
which is needed to enable the self-optimization process.
*/
type Identity struct {
	Name string `json:"name" jsonschema:"title=Name,description=A unique name for the agent,required"`
	Role string `json:"role" jsonschema:"title=Role,description=The role of the agent,required"`
}

/*
NewIdentity attempts to load an existing identity from the datalake, or
creates a new one if it doesn't exist, and then saves it to the datalake.
*/
func NewIdentity(ctx context.Context, role string) *Identity {
	var (
		identity *Identity = &Identity{}
		loaded   *minio.Object
		buf      []byte
		err      error
	)

	if loaded, err = datalake.NewConn().Get(ctx, "identities/"+role); err == nil && loaded != nil {
		if buf, err = io.ReadAll(loaded); err != nil {
			return nil
		}

		if err = json.Unmarshal(buf, identity); err != nil {
			errnie.Error(err)
		}
	} else {
		identity = &Identity{
			Name: utils.NewName(),
			Role: role,
		}

		datalake.NewConn().Put(ctx, "identities/"+role, errnie.SafeMust(func() ([]byte, error) {
			return json.Marshal(identity)
		}), nil)
	}

	return identity
}

func (identity *Identity) String() string {
	return utils.JoinWith(
		"\n",
		fmt.Sprintf("Name: %s", identity.Name),
		fmt.Sprintf("Role: %s", identity.Role),
	)
}
