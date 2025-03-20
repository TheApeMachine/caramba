package browser

import (
	"errors"
	"io"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/rod/lib/proto"
	"github.com/go-rod/stealth"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/fs"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Instance struct {
	buffer *stream.Buffer
}

func NewInstance() *Instance {
	return &Instance{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("browser.Instance.buffer.fn")

			// fp := gogetfp.New(gogetfp.FreeProxyConfig{
			// 	Timeout: 10000,
			// 	Google:  true,
			// })

			// proxy, err := fp.GetWorkingProxy()

			var launch *launcher.Launcher

			// if err != nil || proxy == "" {
			// 	errnie.Error(err)
			// 	errnie.Info("continuing without proxy")
			launch = launcher.New().Headless(false)
			// } else {
			// 	errnie.Info("using proxy", "addr", proxy)
			// 	launch = launcher.New().Headless(false).Set(flags.ProxyServer, proxy)
			// }

			defer launch.Cleanup()

			url, err := launch.Launch()

			if errnie.Error(err) != nil {
				return err
			}

			var browser *rod.Browser

			errnie.Debug("browser.Instance.buffer.fn", "url", url)
			if browser = rod.New().ControlURL(url); browser == nil {
				datura.WithError(errnie.New(
					errnie.WithError(errors.New("failed to create browser")),
				))(artifact)

				return err
			}

			errnie.Debug("browser.Instance.buffer.fn", "browser", browser)
			if err = browser.Connect(); errnie.Error(err) != nil {
				datura.WithError(errnie.New(
					errnie.WithError(err),
				))(artifact)

				return err
			}

			page, err := stealth.Page(browser)

			navurl := datura.GetMetaValue[string](artifact, "url")
			errnie.Debug("browser.Instance.buffer.fn", "navurl", navurl)

			if err = page.Navigate(navurl); errnie.Error(err) != nil {
				datura.WithError(errnie.New(
					errnie.WithError(err),
				))(artifact)

				return err
			}

			op := datura.GetMetaValue[string](artifact, "operation")

			errnie.Debug("browser.Instance.buffer.fn", "op", op)

			if _, err = io.Copy(artifact, workflow.NewPipeline(datura.New(
				datura.WithRole(datura.ArtifactRoleOpenFile),
				datura.WithMeta("path", "scripts/"+op+".js"),
			), fs.NewStore())); err != nil {
				return errnie.Error(err)
			}

			errnie.Debug("browser.Instance.buffer.fn", "status", "decrypting file")

			var payload []byte
			if payload, err = artifact.DecryptPayload(); err != nil {
				return errnie.Error(err)
			}

			var (
				runtime *proto.RuntimeRemoteObject
			)

			if runtime, err = page.Eval(string(payload)); err != nil {
				return errnie.Error(err)
			}

			val := runtime.Value.Get("val").Str()
			browser.Close()

			datura.WithPayload([]byte(utils.SummarizeText(val, 2000)))(artifact)
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
