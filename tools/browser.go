package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"strings"
	"time"

	"github.com/charmbracelet/log"
	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/input"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/rod/lib/proto"
	"github.com/go-rod/stealth"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

type BrowserArgs struct {
	URL        string `json:"url" jsonschema:"required,description=The URL to navigate to"`
	Selector   string `json:"selector" jsonschema:"description=CSS selector to find elements"`
	Timeout    int    `json:"timeout" jsonschema:"description=Timeout in seconds"`
	Screenshot bool   `json:"screenshot" jsonschema:"description=Whether to take a screenshot"`
}

type Browser struct {
	Operation      string   `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=navigate,enum=script"`
	URL            string   `json:"url" jsonschema:"title=URL,description=The URL to navigate to,required"`
	Javascript     string   `json:"javascript" jsonschema:"title=JavaScript,description=JavaScript code to execute in the developer console. Must always be a function that returns a string, example: () => Array.from(document.querySelectorAll('p')).map(p => p.innerText).join('\\n')"`
	ExtractorType  string   `json:"extractor_type" jsonschema:"title=Extractor Type,description=Type of extractor to use (common, site, or custom)"`
	ExtractorName  string   `json:"extractor_name" jsonschema:"title=Extractor Name,description=Name of the pre-made extractor to use"`
	CustomScript   string   `json:"custom_script" jsonschema:"title=Custom Script,description=Custom extraction script to use"`
	HelperNames    []string `json:"helper_names" jsonschema:"title=Helper Names,description=Names of helper functions to include in custom script"`
	instance       *rod.Browser
	page           *rod.Page
	currentElement *rod.Element
	history        []BrowseAction
	proxy          *url.URL
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

// Add new types for enhanced functionality
type BrowserResult struct {
	Content    string `json:"content"`
	Screenshot []byte `json:"screenshot,omitempty"`
	Status     string `json:"status"`
	Error      string `json:"error,omitempty"`
}

func NewBrowser() *Browser {
	return &Browser{
		history: make([]BrowseAction, 0),
	}
}

func (browser *Browser) Name() string {
	return "browser"
}

func (browser *Browser) Description() string {
	return "Interact with the web"
}

func (browser *Browser) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Browser]()
}

func (browser *Browser) Initialize() error {
	return nil
}

func (browser *Browser) Connect(ctx context.Context, conn io.ReadWriteCloser) error {
	return nil
}

func (browser *Browser) Use(ctx context.Context, args map[string]any) string {
	var (
		result string
		err    error
	)

	if result, err = browser.Run(args); err != nil {
		return err.Error()
	}

	return result
}

// SetProxy configures a proxy for the browser
func (browser *Browser) SetProxy(proxyURL string) {
	browser.proxy = errnie.SafeMust(func() (*url.URL, error) {
		return url.Parse(proxyURL)
	})
}

// StartSession initializes a new browsing session with stealth mode
func (browser *Browser) StartSession() error {
	log.Info("Starting browser session")

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
	log.Info("Navigating", "url", url)

	// Instead of creating a new page, use the existing stealth page
	if err := browser.page.Navigate(url); err != nil {
		log.Error("Failed to navigate", "error", err)
		return err
	}

	// Wait for network to be idle and page to be fully loaded
	if err := browser.page.WaitLoad(); err != nil {
		log.Error("Failed to wait for page load", "error", err)
		return err
	}

	// Additional wait for dynamic content
	if err := browser.page.WaitIdle(5); err != nil {
		log.Error("Failed to wait for page idle", "error", err)
		return err
	}

	return nil
}

// ExecuteScript runs custom JavaScript and returns the result
func (browser *Browser) ExecuteScript(script string) string {
	if script == "" {
		log.Warn("No script provided")
		return ""
	}

	if !strings.HasPrefix(script, "()") {
		errnie.Debug("trying to load script", "script", script)
		if utils.ReadFile(utils.Workdir()+"/tools/scripts/"+script+".js") != "" {
			script = utils.ReadFile(utils.Workdir() + "/tools/scripts/" + script + ".js")
		}
	}

	return browser.page.MustEval(script).Str()
}

// TakeScreenshot takes a screenshot of the current page
func (browser *Browser) TakeScreenshot() ([]byte, error) {
	if browser.page == nil {
		return nil, fmt.Errorf("no active page")
	}
	return browser.page.Screenshot(true, &proto.PageCaptureScreenshot{
		Format:  proto.PageCaptureScreenshotFormatPng,
		Quality: utils.IntPtr(100),
		Clip: &proto.PageViewport{
			X:      0,
			Y:      0,
			Width:  1920,
			Height: 1080,
			Scale:  1,
		},
	})
}

// WaitForSelector waits for an element to be visible
func (browser *Browser) WaitForSelector(selector string, timeout int) error {
	if timeout == 0 {
		timeout = 30 // default timeout
	}
	return browser.page.Timeout(time.Duration(timeout) * time.Second).MustElement(selector).WaitVisible()
}

/*
Run the Browser and react to the arguments that were provided by the Large Language Model
*/
func (browser *Browser) Run(args map[string]any) (string, error) {
	result := &BrowserResult{
		Status: "success",
	}

	if proxyURL, ok := args["proxy"].(string); ok {
		browser.SetProxy(proxyURL)
	}

	// Only start a new session if one doesn't exist
	if browser.instance == nil {
		if err := browser.StartSession(); err != nil {
			result.Status = "error"
			result.Error = err.Error()
			return "", err
		}
	}

	// Handle navigation only if URL is provided
	if url, ok := args["url"].(string); ok {
		if err := browser.Navigate(url); err != nil {
			result.Status = "error"
			result.Error = err.Error()
			return "", err
		}
	}

	if script, ok := args["javascript"].(string); ok {
		result.Content = browser.ExecuteScript(script)
	}

	// Handle actions
	if action, ok := args["action"].(string); ok && action != "" {
		if err := browser.handleAction(action); err != nil {
			result.Status = "error"
			result.Error = err.Error()
			return "", err
		}
	}

	// Handle hotkeys
	if hotkeys, ok := args["hotkeys"].(string); ok && hotkeys != "" {
		if err := browser.handleHotkeys(hotkeys); err != nil {
			result.Status = "error"
			result.Error = err.Error()
			return "", err
		}
	}

	// Take screenshot if requested
	if screenshot, ok := args["screenshot"].(bool); ok && screenshot {
		if bytes, err := browser.TakeScreenshot(); err == nil {
			result.Screenshot = bytes
		} else {
			result.Status = "warning"
			result.Error = fmt.Sprintf("screenshot failed: %v", err)
		}
	}

	resultJSON, err := json.Marshal(result)
	if err != nil {
		return "", err
	}

	return string(resultJSON), nil
}

// Helper method to handle actions
func (browser *Browser) handleAction(action string) error {
	if browser.currentElement == nil {
		return fmt.Errorf("no element selected for action: %s", action)
	}

	switch action {
	case "click":
		browser.currentElement.MustClick()
	case "scroll":
		browser.page.Mouse.Scroll(0, 400, 1)
	case "hover":
		browser.currentElement.MustHover()
	case "keypress":
		browser.handleHotkeys(action)
	default:
		return fmt.Errorf("unknown action: %s", action)
	}

	return nil
}

// Helper method to handle hotkeys
func (browser *Browser) handleHotkeys(hotkeys string) error {
	if browser.currentElement == nil {
		return fmt.Errorf("no element selected for hotkeys")
	}

	keys := make([]input.Key, len(hotkeys))
	for i, r := range hotkeys {
		keys[i] = input.Key(r)
	}
	browser.currentElement.MustType(keys...)
	return nil
}

func (browser *Browser) Close() error {
	if browser.instance != nil {
		return browser.instance.Close()
	}
	return nil
}
