# Browser Tool

This directory contains the implementation of the Browser Tool, which provides web browsing capabilities to the agent. The tool is implemented as a client for the Browserless API, which provides a headless Chrome browser as a service.

## Features

The Browser Tool provides the following features:

- **Navigate**: Navigate to a URL and retrieve the page HTML
- **Screenshot**: Take a screenshot of a webpage
- **Extract**: Extract content from a webpage using CSS selectors
- **Search**: Perform web searches using DuckDuckGo
- **PDF**: Generate PDF files from webpages
- **Execute**: Execute custom JavaScript code in the context of a webpage

## Architecture

The code has been refactored from a monolithic design into a more modular structure, with each feature implemented in its own file:

- `browser.go`: Core implementation of the Tool interface, including the main Execute method
- `navigate.go`: Implementation of webpage navigation
- `screenshot.go`: Implementation of screenshot capture
- `extract.go`: Implementation of content extraction using CSS selectors
- `search.go`: Implementation of web search using DuckDuckGo
- `pdf.go`: Implementation of PDF generation
- `execute.go`: Implementation of JavaScript execution
- `utils.go`: Common utility functions used by multiple features

## Usage

The Browser Tool is used through the Agent's tool execution mechanism. It can be called with the following actions:

```go
// Navigate to a URL
result, err := agent.ExecuteTool("browser", map[string]interface{}{
    "action": "navigate",
    "url": "https://example.com",
})

// Take a screenshot
result, err := agent.ExecuteTool("browser", map[string]interface{}{
    "action": "screenshot",
    "url": "https://example.com",
    "full_page": true,
})

// Extract content
result, err := agent.ExecuteTool("browser", map[string]interface{}{
    "action": "extract",
    "url": "https://example.com",
    "selector": ".content",
})

// Search the web
result, err := agent.ExecuteTool("browser", map[string]interface{}{
    "action": "search",
    "query": "How does AI work?",
})

// Generate a PDF
result, err := agent.ExecuteTool("browser", map[string]interface{}{
    "action": "pdf",
    "url": "https://example.com",
    "full_page": true,
})

// Execute JavaScript
result, err := agent.ExecuteTool("browser", map[string]interface{}{
    "action": "execute",
    "url": "https://example.com",
    "script": "return document.title;",
})
```

## Configuration

The Browser Tool is configured using environment variables:

- `BROWSERLESS_API_KEY`: API key for the Browserless service

## Error Handling

All methods include detailed error handling, with errors being propagated up to the caller with appropriate context.

## Future Improvements

- Add caching for frequently accessed pages
- Add rate limiting to avoid API usage limits
- Add more sophisticated content extraction capabilities
- Implement better session management for multi-page interactions
- Add support for more authentication methods
