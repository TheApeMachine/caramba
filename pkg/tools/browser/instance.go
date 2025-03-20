package browser

import (
	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/stealth"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Instance struct {
	buffer *stream.Buffer
}

func NewInstance() *Instance {
	return &Instance{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) error {
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

			page := stealth.MustPage(
				rod.New().ControlURL(
					launch.MustLaunch(),
				).MustConnect(),
			)

			wait := page.MustWaitNavigation()
			page.MustNavigate(datura.GetMetaValue[string](artifact, "url"))
			wait()

			script := datura.GetMetaValue[string](artifact, "script")

			val := page.MustEval(script).Get("val").Str()
			datura.WithPayload([]byte(val[:1000]))(artifact)

			errnie.Info("browser.Instance.buffer.fn", "val", val[:1000])

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
