package browser

import (
	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

// BrowserGenerator implements the stream.Generator interface for browser operations
type BrowserGenerator struct{}

// Generate processes browser operations
func (bg *BrowserGenerator) Generate(in chan datura.Artifact) chan datura.Artifact {
	out := make(chan datura.Artifact)

	go func() {
		defer close(out)

		for artifact := range in {
			errnie.Debug("browser.Instance.buffer.fn")

			manager, err := NewManager(artifact).Initialize()

			if errnie.Error(err) != nil {
				artifact.SetMetaValue("error", err.Error())
				out <- artifact
				continue
			}

			defer manager.Close()

			op := datura.GetMetaValue[string](artifact, "operation")

			switch op {
			case "get_content":
				var (
					content  string
					markdown string
				)

				if content, err = manager.GetPage().HTML(); errnie.Error(err) != nil {
					artifact.SetMetaValue("error", err.Error())
					out <- artifact
					continue
				}

				if markdown, err = htmltomarkdown.ConvertString(content); errnie.Error(err) != nil {
					artifact.SetMetaValue("error", err.Error())
					out <- artifact
					continue
				}

				datura.WithEncryptedPayload([]byte(markdown))(artifact)
			default:
				var val string

				if val, err = NewEval(manager.GetPage(), artifact, op).Run(); errnie.Error(err) != nil {
					artifact.SetMetaValue("error", errnie.Error(err).Error())
					out <- artifact
					continue
				}

				datura.WithEncryptedPayload([]byte(utils.SummarizeText(val, 2000)))(artifact)
			}

			out <- artifact
		}
	}()

	return out
}
