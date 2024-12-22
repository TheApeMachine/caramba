package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"time"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/rod/lib/proto"
	"github.com/go-rod/stealth"
	"github.com/invopop/jsonschema"
	"github.com/spf13/cast"
	"github.com/theapemachine/errnie"
)

type BrowserArgs struct {
	URL        string `json:"url" jsonschema:"required,description=The URL to navigate to"`
	Selector   string `json:"selector" jsonschema:"description=CSS selector to find elements"`
	Timeout    int    `json:"timeout" jsonschema:"description=Timeout in seconds"`
	Screenshot bool   `json:"screenshot" jsonschema:"description=Whether to take a screenshot"`
}

type Browser struct {
	Operation   string            `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=navigate,enum=click,enum=extract,enum=script,enum=wait,enum=form,enum=screenshot,enum=intercept,enum=cookies,enum=hijack,enum=response,enum=close"`
	URL         string            `json:"url" jsonschema:"title=URL,description=The URL to navigate to,required"`
	GoogleDorks []string          `json:"google_dorks" jsonschema:"title=Google Dorks,description=Google Dorks to use for research enabling highly specific results"`
	Selector    string            `json:"selector" jsonschema:"title=Selector,description=CSS selector to find elements"`
	Javascript  string            `json:"javascript" jsonschema:"title=JavaScript,description=JavaScript code to execute in the developer console"`
	Hijack      string            `json:"hijack" jsonschema:"title=Hijack,description=Hijack a network request"`
	Response    string            `json:"response" jsonschema:"title=Response,description=Response to return for a network request"`
	Form        map[string]string `json:"form" jsonschema:"title=Form,description=Form data to fill in"`
	Intercept   []string          `json:"intercept" jsonschema:"title=Intercept,description=Network intercept patterns"`
	Cookies     string            `json:"cookies" jsonschema:"title=Cookies,description=Cookie operation,enum=get,enum=set,enum=delete"`
	instance    *rod.Browser
	page        *rod.Page
	history     []BrowseAction
	proxy       *url.URL
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

func (browser *Browser) Use(ctx context.Context, args map[string]any) string {
	return errnie.SafeMust(func() (string, error) {
		return browser.Run(args)
	})
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
func (browser *Browser) StartSession() {
	l := launcher.New().
		Headless(false).
		Set("disable-web-security", "").
		Set("disable-setuid-sandbox", "").
		Set("no-sandbox", "")

	if browser.proxy != nil {
		l.Proxy(browser.proxy.String())
	}

	url := errnie.SafeMust(func() (string, error) {
		return l.Launch()
	})

	browser.instance = rod.New().
		ControlURL(url).
		MustConnect()

	// Create a new stealth page instead of regular page
	browser.page = errnie.SafeMust(func() (*rod.Page, error) {
		return stealth.Page(browser.instance)
	})

	browser.instance.MustIgnoreCertErrors(true)
}

// Navigate goes to a URL and waits for the page to load
func (browser *Browser) Navigate(url string) {
	// Instead of creating a new page, use the existing stealth page
	errnie.MustVoid(browser.page.Navigate(url))
	errnie.MustVoid(browser.page.WaitLoad())
	browser.recordAction("navigate", url, "", true)
}

// Click finds and clicks an element using various selectors
func (browser *Browser) Click(selector string) {
	el := errnie.SafeMust(func() (*rod.Element, error) {
		return browser.page.Element(selector)
	})

	errnie.MustVoid(el.Click(proto.InputMouseButtonLeft, 1))
	browser.recordAction("click", selector, "", true)
}

// Extract gets content from the page using a CSS selector
func (browser *Browser) Extract(selector string) string {
	el := errnie.SafeMust(func() (*rod.Element, error) {
		return browser.page.Element(selector)
	})

	text := errnie.SafeMust(func() (string, error) {
		return el.Text()
	})

	browser.recordAction("extract", selector, text, true)
	return text
}

// ExecuteScript runs custom JavaScript and returns the result
func (browser *Browser) ExecuteScript(script string) interface{} {
	if script == "" {
		return nil
	}

	result := errnie.SafeMust(func() (interface{}, error) {
		return browser.page.Eval(script)
	})

	browser.recordAction("script", script, fmt.Sprintf("%v", result), true)
	return result
}

// WaitForElement waits for an element to appear
func (browser *Browser) WaitForElement(selector string, timeout time.Duration) error {
	// Remove unused context
	errnie.MustVoid(browser.page.Timeout(timeout).MustElement(selector).WaitVisible())
	browser.recordAction("wait", selector, "", true)
	return nil
}

// GetHistory returns the browsing session history
func (browser *Browser) GetHistory() []BrowseAction {
	return browser.history
}

// Run implements the enhanced interface with all new capabilities
func (browser *Browser) Run(args map[string]any) (string, error) {
	if proxyURL, ok := args["proxy"].(string); ok {
		browser.SetProxy(proxyURL)
	}

	// Only start a new session if one doesn't exist
	if browser.instance == nil {
		browser.StartSession()
	}

	// Remove the defer close
	// defer browser.instance.Close()

	// Handle navigation only if URL is provided
	if url, ok := args["url"].(string); ok {
		browser.Navigate(url)
	}

	// Handle form filling
	if formData, ok := args["form"].(map[string]string); ok {
		browser.FillForm(formData)
	}

	// Handle screenshots
	if screenshot, ok := args["screenshot"].(map[string]string); ok {
		browser.Screenshot(screenshot["selector"], screenshot["filepath"])
	}

	// Handle network interception
	if patterns, ok := args["intercept"].([]string); ok {
		browser.InterceptNetwork(patterns)
	}

	// Handle cookie operations
	if cookieOp, ok := args["cookies"].(string); ok {
		switch cookieOp {
		case "get":
			browser.ManageCookies()
		case "set":
			if cookie, ok := args["cookie"].(Cookie); ok {
				browser.SetCookie(cookie)
			}
		case "delete":
			if cookieData, ok := args["cookie_data"].(map[string]string); ok {
				browser.DeleteCookies(cookieData["name"], cookieData["domain"])
			}
		}
	}

	// Continue with existing functionality...
	if script, ok := args["javascript"].(string); ok && script != "" {
		browser.ExecuteScript(script)
	}

	return browser.page.MustInfo().Title, nil
}

// FillForm fills form fields with provided data
func (browser *Browser) FillForm(fields map[string]string) {
	for selector, value := range fields {
		el := errnie.SafeMust(func() (*rod.Element, error) {
			return browser.page.Element(selector)
		})

		// Clear existing value
		el.MustEval(`el => el.value = ''`)

		// Input new value
		errnie.MustVoid(el.Input(value))

		browser.recordAction("fill_form", map[string]string{
			"selector": selector,
			"value":    value,
		}, "", true)
	}
}

// Screenshot captures the current page or element
func (browser *Browser) Screenshot(selector string, filepath string) {
	var img []byte

	if selector == "" {
		// Capture full page
		img = errnie.SafeMust(func() ([]byte, error) {
			return browser.page.Screenshot(true, &proto.PageCaptureScreenshot{
				Format:      proto.PageCaptureScreenshotFormatPng,
				FromSurface: true,
			})
		})
	} else {
		// Capture specific element
		el := errnie.SafeMust(func() (*rod.Element, error) {
			return browser.page.Element(selector)
		})
		img = errnie.SafeMust(func() ([]byte, error) {
			return el.Screenshot(proto.PageCaptureScreenshotFormatPng, 1)
		})
	}

	errnie.MustVoid(os.WriteFile(filepath, img, 0644))
	browser.recordAction("screenshot", map[string]string{
		"selector": selector,
		"filepath": filepath,
	}, "", true)
}

// InterceptNetwork starts intercepting network requests
func (browser *Browser) InterceptNetwork(patterns []string) error {
	router := browser.page.HijackRequests()
	defer router.Stop()

	for _, pattern := range patterns {
		router.MustAdd(pattern, func(ctx *rod.Hijack) {
			// Fix headers type conversion
			headers := make(map[string]string)
			for k, v := range ctx.Request.Headers() {
				headers[k] = v.String() // Convert gson.JSON to string
			}

			// Fix body type conversion
			body := ctx.Request.Body() // Call the function to get the body string

			req := NetworkRequest{
				URL:     ctx.Request.URL().String(),
				Method:  ctx.Request.Method(),
				Headers: headers,
				Body:    body,
			}

			browser.recordAction("network_request", req, "", true)
			ctx.ContinueRequest(&proto.FetchContinueRequest{})
		})
	}

	return nil
}

// ManageCookies provides cookie management capabilities
func (browser *Browser) ManageCookies() []Cookie {
	cookies := errnie.SafeMust(func() ([]*proto.NetworkCookie, error) {
		return browser.page.Cookies([]string{})
	})

	var result []Cookie
	for _, c := range cookies {
		cookie := Cookie{
			Name:     c.Name,
			Value:    c.Value,
			Domain:   c.Domain,
			Path:     c.Path,
			Expires:  time.Unix(int64(c.Expires), 0),
			Secure:   c.Secure,
			HTTPOnly: c.HTTPOnly,
		}
		result = append(result, cookie)
	}

	browser.recordAction("get_cookies", nil, fmt.Sprintf("%d cookies", len(result)), true)
	return result
}

// SetCookie adds a new cookie
func (browser *Browser) SetCookie(cookie Cookie) {
	// Fix SetCookies argument type
	errnie.MustVoid(browser.page.SetCookies([]*proto.NetworkCookieParam{
		{
			Name:     cookie.Name,
			Value:    cookie.Value,
			Domain:   cookie.Domain,
			Path:     cookie.Path,
			Expires:  proto.TimeSinceEpoch(cookie.Expires.Unix()),
			Secure:   cookie.Secure,
			HTTPOnly: cookie.HTTPOnly,
		},
	}))

	browser.recordAction("set_cookie", cookie, "", true)
}

// DeleteCookies removes cookies matching the given parameters
func (browser *Browser) DeleteCookies(name, domain string) {
	// Fix non-variadic function call
	errnie.MustVoid(browser.page.SetCookies([]*proto.NetworkCookieParam{
		{
			Name:   name,
			Domain: domain,
		},
	}))

	browser.recordAction("delete_cookies", map[string]string{
		"name":   name,
		"domain": domain,
	}, "", true)
}

func (browser *Browser) recordAction(actionType string, data interface{}, result string, success bool) {
	browser.history = append(browser.history, BrowseAction{
		Type:    actionType,
		Data:    data,
		Result:  result,
		Time:    time.Now(),
		Success: success,
	})
}

func (browser *Browser) Close() error {
	if browser.instance != nil {
		return browser.instance.Close()
	}
	return nil
}

// Add the Execute method to implement the Tool interface
func (browser *Browser) Execute(ctx context.Context, args map[string]interface{}) string {
	// Get URL from args
	url := errnie.SafeMust(func() (string, error) {
		return getStringArg(args, "url", "")
	})

	// Navigate to URL
	browser.Navigate(url)

	// Get selector if provided, default to "body"
	selector := errnie.SafeMust(func() (string, error) {
		return getStringArg(args, "selector", "body")
	})

	// Wait for element if timeout provided
	if timeout, ok := args["timeout"].(float64); ok {
		browser.WaitForElement(selector, time.Duration(timeout)*time.Second)
	}

	// Extract content
	return browser.Extract(selector)
}

func getStringArg(args map[string]interface{}, key string, defaultValue string) (string, error) {
	value, ok := args[key]
	if !ok {
		return defaultValue, nil
	}
	return cast.ToString(value), nil
}

// Add a new method to explicitly close the browser when needed
func (browser *Browser) CleanupSession() {
	if browser.instance != nil {
		browser.instance.Close()
		browser.instance = nil
		browser.page = nil
	}
}
