# Examples

This section provides practical examples of how to interact with a running Caramba agent using the A2A protocol from a Go application. These examples assume Caramba is running and accessible at `http://localhost:8080`.

**Note:** These examples focus on the client-side interaction using Go's `net/http` package to send JSON-RPC requests. They replace the previous examples that used outdated internal Caramba patterns.

## Core Interaction Pattern

The basic flow for interacting with a Caramba agent via A2A involves:

1. **`TaskCreate`:** Send a request to the `/rpc` endpoint with method `TaskCreate` to initialize a task and get a unique `task_id`.
2. **`TaskSend`:** Send a subsequent request to `/rpc` with method `TaskSend`, providing the `task_id` and the actual user message/prompt.
3. **(Optional) Subscribe to Stream:** Connect to the `/task/:id/stream` endpoint using an SSE client to receive real-time updates.
4. **Process Response:** Handle the initial JSON-RPC response from `TaskSend` (confirming acceptance) and process the events received on the stream (or wait for a final status notification if not streaming).

## Example 1: Simple Question

Let's ask the agent a basic question.

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net/http"
    "time"
)

const (
    carambaURL = "http://localhost:8080" // Adjust if your agent runs elsewhere
    rpcPath    = "/rpc"
)

// Define basic JSON-RPC request/response structures
type JsonRpcRequest struct {
    JsonRpc string      `json:"jsonrpc"`
    Method  string      `json:"method"`
    Params  interface{} `json:"params"`
    ID      string      `json:"id"`
}

type JsonRpcResponse struct {
    JsonRpc string      `json:"jsonrpc"`
    Result  interface{} `json:"result,omitempty"`
    Error   interface{} `json:"error,omitempty"`
    ID      string      `json:"id"`
}

// Define A2A specific parameter structures
type TaskCreateParams struct {
    TaskID       string      `json:"task_id"`
    Input        interface{} `json:"input,omitempty"` // Define detailed Input struct if needed
    AgentCardURL string      `json:"agent_card_url,omitempty"`
}

type MessagePart struct {
    Type string `json:"type"`
    Text string `json:"text"`
}

type Message struct {
    RequestID string        `json:"request_id,omitempty"`
    Role      string        `json:"role"`
    Parts     []MessagePart `json:"parts"`
    Metadata  interface{}   `json:"metadata,omitempty"`
}

type TaskSendParams struct {
    TaskID    string  `json:"task_id"`
    Message   Message `json:"message"`
    Subscribe bool    `json:"subscribe"`
}

// Helper to send JSON-RPC requests
func sendRpcRequest(url, method string, params interface{}, id string) (*JsonRpcResponse, error) {
    reqBody := JsonRpcRequest{
        JsonRpc: "2.0",
        Method:  method,
        Params:  params,
        ID:      id,
    }

    jsonData, err := json.Marshal(reqBody)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }

    httpResp, err := http.Post(url+rpcPath, "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, fmt.Errorf("http post failed: %w", err)
    }
    defer httpResp.Body.Close()

    if httpResp.StatusCode != http.StatusOK {
        bodyBytes, _ := io.ReadAll(httpResp.Body)
        return nil, fmt.Errorf("http request failed: status=%d, body=%s", httpResp.StatusCode, string(bodyBytes))
    }

    var respBody JsonRpcResponse
    if err := json.NewDecoder(httpResp.Body).Decode(&respBody); err != nil {
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }

    if respBody.Error != nil {
        log.Printf("RPC Error received: %+v", respBody.Error)
        // Return the response containing the error, let caller handle it
        return &respBody, nil
    }

    return &respBody, nil
}

func main() {
    taskID := fmt.Sprintf("go-client-task-%d", time.Now().UnixNano())
    requestIDBase := fmt.Sprintf("go-client-req-%d", time.Now().UnixNano())

    // 1. Create the Task
    log.Printf("Creating task: %s\n", taskID)
    taskCreateParams := TaskCreateParams{
        TaskID: taskID,
        // Input can be minimal here if the main prompt is in TaskSend
    }
    createResp, err := sendRpcRequest(carambaURL, "TaskCreate", taskCreateParams, requestIDBase+"-create")
    if err != nil {
        log.Fatalf("TaskCreate failed: %v", err)
    }
    if createResp.Error != nil {
        log.Fatalf("TaskCreate RPC error: %+v", createResp.Error)
    }
    log.Printf("TaskCreate Result: %+v\n", createResp.Result)

    // --- Optional: Start SSE listener here in a goroutine ---
    // See Example 3 for SSE handling

    // 2. Send the Message
    log.Printf("Sending message to task: %s\n", taskID)
    userMessage := Message{
        Role: "user",
        Parts: []MessagePart{
            {Type: "text", Text: "What is the population of Tokyo?"},
        },
        RequestID: requestIDBase + "-send-1",
    }
    taskSendParams := TaskSendParams{
        TaskID:    taskID,
        Message:   userMessage,
        Subscribe: true, // Request streaming updates
    }
    sendResp, err := sendRpcRequest(carambaURL, "TaskSend", taskSendParams, requestIDBase+"-send")
    if err != nil {
        log.Fatalf("TaskSend failed: %v", err)
    }
    if sendResp.Error != nil {
        log.Fatalf("TaskSend RPC error: %+v", sendResp.Error)
    }
    log.Printf("TaskSend initial result: %+v\n", sendResp.Result)

    // Since we requested subscribe: true, the actual answer will arrive
    // via the SSE stream or a final notification.
    // Without an SSE listener, this example only shows the initial setup.
    log.Println("Task initiated. Listen on SSE stream for the full response.")

    // Keep main running for a bit if you have an SSE listener in a goroutine
    // time.Sleep(30 * time.Second)
}
```

## Example 2: Using a Tool (Web Research)

This example asks the agent to research a topic, which likely requires the `Browser` tool.

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net/http"
    "time"
)

const (
    carambaURL = "http://localhost:8080"
    rpcPath    = "/rpc"
)

func main() {
    taskID := fmt.Sprintf("go-research-task-%d", time.Now().UnixNano())
    requestIDBase := fmt.Sprintf("go-research-req-%d", time.Now().UnixNano())

    // 1. Create Task
    log.Printf("Creating task: %s\n", taskID)
    taskCreateParams := TaskCreateParams{ TaskID: taskID }
    createResp, err := sendRpcRequest(carambaURL, "TaskCreate", taskCreateParams, requestIDBase+"-create")
    if err != nil || createResp.Error != nil {
        log.Fatalf("TaskCreate failed. Err: %v, RPC Err: %+v", err, createResp.Error)
    }
    log.Printf("TaskCreate Result: %+v\n", createResp.Result)

    // 2. Send Research Request
    log.Printf("Sending research request to task: %s\n", taskID)
    userMessage := Message{
        Role: "user",
        Parts: []MessagePart{
            {Type: "text", Text: "What are the main features of the Go programming language? Use web search if needed."},
        },
        RequestID: requestIDBase + "-send-1",
    }
    taskSendParams := TaskSendParams{
        TaskID:    taskID,
        Message:   userMessage,
        Subscribe: true,
    }
    sendResp, err := sendRpcRequest(carambaURL, "TaskSend", taskSendParams, requestIDBase+"-send")
    if err != nil || sendResp.Error != nil {
        log.Fatalf("TaskSend failed. Err: %v, RPC Err: %+v", err, sendResp.Error)
    }
    log.Printf("TaskSend initial result: %+v\n", sendResp.Result)

    log.Println("Research task initiated. Listen on SSE stream for progress and results (including potential tool calls).")

    // Add SSE listener (like in Example 3) to see the full interaction
}
```

## Example 3: Handling SSE Stream

This example shows how to connect to the SSE stream for a task and process events.

```go
package main

import (
    "bufio"
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net/http"
    "strings"
    "time"
)

const (
    carambaURL = "http://localhost:8080"
    rpcPath    = "/rpc"
    streamPath = "/task/%s/stream"
)

// Function to listen to SSE stream
func listenToSSE(taskID string) {
    streamURL := fmt.Sprintf(carambaURL+streamPath, taskID)
    log.Printf("Connecting to SSE stream: %s\n", streamURL)

    req, err := http.NewRequest("GET", streamURL, nil)
    if err != nil {
        log.Printf("Error creating SSE request: %v", err)
        return
    }
    req.Header.Set("Accept", "text/event-stream")
    req.Header.Set("Cache-Control", "no-cache")
    req.Header.Set("Connection", "keep-alive")

    client := &http.Client{Timeout: 0} // No timeout for long-lived connection
    resp, err := client.Do(req)
    if err != nil {
        log.Printf("Error connecting to SSE stream: %v", err)
        return
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        log.Printf("SSE connection failed with status: %s", resp.Status)
        return
    }

    log.Println("SSE connection established. Reading events...")
    reader := bufio.NewReader(resp.Body)
    var eventType, eventData string

    for {
        line, err := reader.ReadString('\n')
        if err != nil {
            if err == io.EOF {
                log.Println("SSE stream closed by server.")
            } else {
                log.Printf("Error reading SSE stream: %v", err)
            }
            return // Exit loop on error or EOF
        }

        line = strings.TrimSpace(line)

        if strings.HasPrefix(line, "event:") {
            eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
        } else if strings.HasPrefix(line, "data:") {
            // Accumulate data if it spans multiple lines (though usually not needed for simple JSON)
            eventData += strings.TrimSpace(strings.TrimPrefix(line, "data:"))
        } else if line == "" {
            // Empty line signifies end of event
            if eventType != "" || eventData != "" {
                log.Printf("--- SSE Event ---")
                log.Printf("  Type: %s\n", eventType)
                log.Printf("  Data: %s\n", eventData)
                log.Printf("-----------------")
                // TODO: Parse eventData JSON based on eventType for more structured handling
            }
            // Reset for next event
            eventType = ""
            eventData = ""
        }
        // Ignore other lines (like comments starting with ":")
    }
}

func main() {
    taskID := fmt.Sprintf("go-sse-task-%d", time.Now().UnixNano())
    requestIDBase := fmt.Sprintf("go-sse-req-%d", time.Now().UnixNano())

    // 1. Create Task
    log.Printf("Creating task: %s\n", taskID)
    taskCreateParams := TaskCreateParams{ TaskID: taskID }
    createResp, err := sendRpcRequest(carambaURL, "TaskCreate", taskCreateParams, requestIDBase+"-create")
    if err != nil || createResp.Error != nil {
        log.Fatalf("TaskCreate failed. Err: %v, RPC Err: %+v", err, createResp.Error)
    }
    log.Printf("TaskCreate Result: %+v\n", createResp.Result)

    // 2. Start SSE Listener in a Goroutine
    go listenToSSE(taskID)
    // Give the listener a moment to connect (optional)
    time.Sleep(500 * time.Millisecond)

    // 3. Send the Message
    log.Printf("Sending message to task: %s\n", taskID)
    userMessage := Message{
        Role: "user",
        Parts: []MessagePart{
            {Type: "text", Text: "Tell me a joke about computers."},
        },
        RequestID: requestIDBase + "-send-1",
    }
    taskSendParams := TaskSendParams{
        TaskID:    taskID,
        Message:   userMessage,
        Subscribe: true, // Crucial for getting SSE events
    }
    sendResp, err := sendRpcRequest(carambaURL, "TaskSend", taskSendParams, requestIDBase+"-send")
    if err != nil || sendResp.Error != nil {
        // Log fatal but allow SSE listener to potentially run longer
        log.Printf("TaskSend failed. Err: %v, RPC Err: %+v", err, sendResp.Error)
    } else {
        log.Printf("TaskSend initial result: %+v\n", sendResp.Result)
    }

    log.Println("Task initiated. SSE listener is active.")

    // Keep the main function alive to allow the SSE listener to receive events
    // In a real application, you might use channels or wait groups
    // to manage the lifecycle based on SSE events (e.g., wait for a final event).
    time.Sleep(60 * time.Second) // Keep alive for 60 seconds
    log.Println("Example finished.")
}
```
