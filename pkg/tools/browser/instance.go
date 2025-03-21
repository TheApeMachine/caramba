package browser

import (
	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
)

type Instance struct {
	buffer *stream.Buffer
}

func NewInstance() *Instance {
	return &Instance{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("browser.Instance.buffer.fn")

			manager, err := NewManager(artifact).Initialize()

			if errnie.Error(err) != nil {
				return err
			}

			defer manager.Close()

			op := datura.GetMetaValue[string](artifact, "operation")

			switch op {
			case "get_content":
				var (
					content  string
					markdown string
				)

				if content, err = manager.page.HTML(); errnie.Error(err) != nil {
					return err
				}

				if markdown, err = htmltomarkdown.ConvertString(content); errnie.Error(err) != nil {
					return err
				}

				datura.WithPayload([]byte(utils.SummarizeText(markdown, 2000)))(artifact)
			default:
				var val string

				if val, err = NewEval(manager.page, artifact, op).Run(); errnie.Error(err) != nil {
					return errnie.Error(err)
				}

				datura.WithPayload([]byte(utils.SummarizeText(val, 2000)))(artifact)
			}
			return nil
		}),
	}
}

func (instance *Instance) Read(p []byte) (n int, err error) {
	errnie.Debug("browser.Instance.Read")
	return instance.buffer.Read(p)
}

func (instance *Instance) Write(p []byte) (n int, err error) {
	errnie.Debug("browser.Instance.Write")
	return instance.buffer.Write(p)
}

func (instance *Instance) Close() error {
	errnie.Debug("browser.Instance.Close")
	return nil
}
