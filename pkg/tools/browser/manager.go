package browser

import (
	"errors"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/stealth"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Manager struct {
	launch   *launcher.Launcher
	browser  *rod.Browser
	page     *rod.Page
	artifact *datura.Artifact
}

func NewManager(artifact *datura.Artifact) *Manager {
	return &Manager{artifact: artifact}
}

func (manager *Manager) Initialize() (*Manager, error) {
	// fp := gogetfp.New(gogetfp.FreeProxyConfig{
	// 	Timeout: 10000,
	// 	Google:  true,
	// })

	// proxy, err := fp.GetWorkingProxy()

	// if err != nil || proxy == "" {
	// 	errnie.Error(err)
	// 	errnie.Info("continuing without proxy")
	manager.launch = launcher.New().Headless(false)
	// } else {
	// 	errnie.Info("using proxy", "addr", proxy)
	// 	launch = launcher.New().Headless(false).Set(flags.ProxyServer, proxy)
	// }

	url, err := manager.launch.Launch()

	if errnie.Error(err) != nil {
		return manager, err
	}

	if manager.browser = rod.New().ControlURL(url); manager.browser == nil {
		return manager, manager.artifact.Error(errors.New("failed to create browser"))
	}

	if err = manager.browser.Connect(); err != nil {
		return manager, manager.artifact.Error(err)
	}

	if manager.page, err = stealth.Page(manager.browser); err != nil {
		return manager, manager.artifact.Error(err)
	}

	navurl := datura.GetMetaValue[string](manager.artifact, "url")
	errnie.Debug("browser.Instance.buffer.fn", "navurl", navurl)

	if err = manager.page.Navigate(navurl); errnie.Error(err) != nil {
		return manager, manager.artifact.Error(err)
	}

	return manager, nil
}

func (manager *Manager) Close() (err error) {
	manager.launch.Cleanup()
	return manager.browser.Close()
}
