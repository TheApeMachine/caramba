package browser

import (
	"io"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/proto"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/fs"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Eval struct {
	page     *rod.Page
	artifact *datura.Artifact
	op       string
}

func NewEval(page *rod.Page, artifact *datura.Artifact, op string) *Eval {
	return &Eval{page: page, artifact: artifact, op: op}
}

func (eval *Eval) Run() (result string, err error) {
	if _, err = io.Copy(eval.artifact, workflow.NewPipeline(datura.New(
		datura.WithRole(datura.ArtifactRoleOpenFile),
		datura.WithMeta("path", "scripts/"+eval.op+".js"),
	), fs.NewStore())); err != nil {
		return err.Error(), errnie.Error(err)
	}

	errnie.Debug("browser.Instance.buffer.fn", "status", "decrypting file")

	var payload []byte
	if payload, err = eval.artifact.DecryptPayload(); err != nil {
		return err.Error(), errnie.Error(err)
	}

	var (
		runtime *proto.RuntimeRemoteObject
	)

	if runtime, err = eval.page.Eval(string(payload)); err != nil {
		return err.Error(), errnie.Error(err)
	}

	val := runtime.Value.Get("val").Str()

	return val, nil
}
