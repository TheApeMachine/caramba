package ai

import (
	"context"
	"encoding/json"
	"fmt"

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
func NewIdentity(ctx context.Context, role string) (identity *Identity) {
	identity = &Identity{}

	if loaded := errnie.SafeMust(func() (*minio.Object, error) {
		return datalake.NewConn().Get(ctx, "identities/"+role)
	}); loaded != nil {
		errnie.SafeMustVoid(func() error {
			return json.NewDecoder(loaded).Decode(identity)
		})

		if identity.Name == "" {
			identity = &Identity{
				Name: utils.NewName(),
				Role: role,
			}

			datalake.NewConn().Put(ctx, "identities/"+role, errnie.SafeMust(func() ([]byte, error) {
				return json.Marshal(identity)
			}), nil)

			errnie.Info("Created new identity: %s (%s)", identity.Name, identity.Role)
			return
		}

		errnie.Info("Loaded identity: %s (%s)", identity.Name, identity.Role)
	}

	return
}

func (identity *Identity) String() string {
	return utils.JoinWith(
		"\n",
		"\t<identity>",
		fmt.Sprintf("\t\tName: %s", identity.Name),
		fmt.Sprintf("\t\tRole: %s", identity.Role),
		"\t</identity>",
	)
}
