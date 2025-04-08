package browser

import (
	"io"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/proto"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/fs"
)

type Eval struct {
	page     *rod.Page
	artifact datura.Artifact
	op       string
}

func NewEval(page *rod.Page, artifact datura.Artifact, op string) *Eval {
	return &Eval{page: page, artifact: artifact, op: op}
}

func (eval *Eval) Run() (result string, err error) {
	// Create a script artifact
	scriptArtifact := datura.New(
		datura.WithRole(datura.ArtifactRoleOpenFile),
		datura.WithMeta("path", "scripts/"+eval.op+".js"),
	)

	// Create store and input channel
	store := fs.NewStore()
	inputChan := make(chan datura.Artifact, 1)
	inputChan <- scriptArtifact
	close(inputChan)

	// Get output from store
	outputChan := store.Generate(inputChan)
	scriptContent := <-outputChan

	// Copy script content to the artifact
	if _, err = io.Copy(eval.artifact, scriptContent); err != nil {
		return err.Error(), errnie.Error(err)
	}

	errnie.Debug("browser.Eval.Run", "status", "decrypting file")

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
	errnie.Debug("browser.Eval.Run", "val", val)

	return val, nil
}
