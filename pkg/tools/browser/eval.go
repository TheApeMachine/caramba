package browser

import (
	"fmt"
	"io"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/proto"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
	fs "github.com/theapemachine/caramba/pkg/stores/fs"
)

type Eval struct {
	page     *rod.Page
	op       string
	fsStore  *fs.Store
	toolcall mcp.CallToolRequest
}

func NewEval(page *rod.Page, fsStore *fs.Store, toolcall mcp.CallToolRequest) *Eval {
	op := toolcall.Request.Method
	return &Eval{
		page:     page,
		op:       op,
		fsStore:  fsStore,
		toolcall: toolcall,
	}
}

func (eval *Eval) Run() (result string, err error) {
	var (
		runtime *proto.RuntimeRemoteObject
	)

	scriptPath := fmt.Sprintf("scripts/%s.js", eval.op)

	scriptFile, err := eval.fsStore.Get(scriptPath)
	if err != nil {
		return "", errnie.Error(fmt.Errorf("failed to get script '%s': %w", scriptPath, err))
	}
	defer scriptFile.Close()

	scriptContentBytes, err := io.ReadAll(scriptFile)
	if err != nil {
		return "", errnie.Error(fmt.Errorf("failed to read script '%s': %w", scriptPath, err))
	}
	scriptContent := string(scriptContentBytes)

	if runtime, err = eval.page.Eval(scriptContent); err != nil {
		return err.Error(), errnie.Error(err)
	}

	val := runtime.Value.Get("val").Str()
	errnie.Debug("browser.Eval.Run", "val", val)

	return val, nil
}
