package tools

import (
	"encoding/json"
	"io"
	"net/url"
	"time"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/stealth"
	"github.com/invopop/jsonschema"
	"github.com/theapemachine/errnie"
)

type BrowserArgs struct {
	URL        string `json:"url" jsonschema:"required,description=The URL to navigate to"`
	Selector   string `json:"selector" jsonschema:"description=CSS selector to find elements"`
	Timeout    int    `json:"timeout" jsonschema:"description=Timeout in seconds"`
	Screenshot bool   `json:"screenshot" jsonschema:"description=Whether to take a screenshot"`
}

type Browser struct {
	Operation  string `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=navigate,enum=script"`
	URL        string `json:"url" jsonschema:"title=URL,description=The URL to navigate to,required"`
	Javascript string `json:"javascript" jsonschema:"title=JavaScript,description=JavaScript code to execute in the developer console. Must always be a function that returns a string, example: () => Array.from(document.querySelectorAll('p')).map(p => p.innerText).join('\\n')"`
	instance   *rod.Browser
	page       *rod.Page
	history    []BrowseAction
	proxy      *url.URL
}

type BrowseAction struct {
	Type    string      `json:"type"`
	Data    interface{} `json:"data"`
	Result  string      `json:"result"`
	Time    time.Time   `json:"time"`
	Success bool        `json:"success"`
}

type NetworkRequest struct {
	URL     string            `json:"url"`
	Method  string            `json:"method"`
	Headers map[string]string `json:"headers"`
	Body    string            `json:"body"`
}

type Cookie struct {
	Name     string    `json:"name"`
	Value    string    `json:"value"`
	Domain   string    `json:"domain"`
	Path     string    `json:"path"`
	Expires  time.Time `json:"expires"`
	Secure   bool      `json:"secure"`
	HTTPOnly bool      `json:"http_only"`
}

func NewBrowser() *Browser {
	return &Browser{
		history: make([]BrowseAction, 0),
	}
}

func (browser *Browser) Initialize() error {
	return nil
}

func (browser *Browser) Connect(conn io.ReadWriteCloser) error {
	return nil
}

func (browser *Browser) Use(args map[string]any) string {
	var (
		result string
		err    error
	)

	if result, err = browser.Run(args); err != nil {
		return err.Error()
	}

	return result
}

func (browser *Browser) GenerateSchema() string {
	return string(errnie.SafeMust(func() ([]byte, error) {
		schema := jsonschema.Reflect(&Browser{})
		return json.MarshalIndent(schema, "", "  ")
	}))
}

// SetProxy configures a proxy for the browser
func (browser *Browser) SetProxy(proxyURL string) {
	browser.proxy = errnie.SafeMust(func() (*url.URL, error) {
		return url.Parse(proxyURL)
	})
}

// StartSession initializes a new browsing session with stealth mode
func (browser *Browser) StartSession() error {
	l := launcher.New().
		Headless(false).
		Set("disable-web-security", "").
		Set("disable-setuid-sandbox", "").
		Set("no-sandbox", "")

	if browser.proxy != nil {
		l.Proxy(browser.proxy.String())
	}

	debugURL, err := l.Launch()

	if err != nil {
		return err
	}

	browser.instance = rod.New().
		ControlURL(debugURL).
		MustConnect()

	// Create a new stealth page instead of regular page
	browser.page, err = stealth.Page(browser.instance)

	if err != nil {
		return err
	}

	browser.instance.MustIgnoreCertErrors(true)

	return nil
}

// Navigate goes to a URL and waits for the page to load
func (browser *Browser) Navigate(url string) error {
	// Instead of creating a new page, use the existing stealth page
	if err := browser.page.Navigate(url); err != nil {
		return err
	}

	if err := browser.page.WaitLoad(); err != nil {
		return err
	}

	return nil
}

// ExecuteScript runs custom JavaScript and returns the result
func (browser *Browser) ExecuteScript(script string) string {
	if script == "" {
		return ""
	}

	result, err := browser.page.Eval(script)

	if err != nil {
		return err.Error()
	}

	return result.Value.Str()
}

// Run implements the enhanced interface with all new capabilities
func (browser *Browser) Run(args map[string]any) (string, error) {
	if proxyURL, ok := args["proxy"].(string); ok {
		browser.SetProxy(proxyURL)
	}

	// Only start a new session if one doesn't exist
	if browser.instance == nil {
		if err := browser.StartSession(); err != nil {
			return "", err
		}
	}

	// Handle navigation only if URL is provided
	if url, ok := args["url"].(string); ok {
		if err := browser.Navigate(url); err != nil {
			return "", err
		}
	}

	// Continue with existing functionality...
	if script, ok := args["javascript"].(string); ok && script != "" {
		return browser.ExecuteScript(script), nil
	}

	return browser.page.MustInfo().Title, nil
}

func (browser *Browser) Close() error {
	if browser.instance != nil {
		return browser.instance.Close()
	}
	return nil
}
