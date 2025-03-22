# NOTES

Keep any important information, goals, insights, etc. by writing them down in this file.

It will be attached to each request so you will have access to it.

## Code Examples of Issues

### 1. Race Conditions in Buffer Access

In `pkg/tools/environment/runner.go`:

```go
go func() {
    buf := make([]byte, 1024)
    for {
        n, err = runner.bufOut.Read(buf)  // <- Concurrent access without mutex
        if n == 0 || err != nil {
            select {
            case runner.outputReady <- struct{}{}:
            default:
            }
            continue
        }
    }
}()
```

Should use proper synchronization:

```go
type Runner struct {
    bufMu sync.Mutex
    // ... other fields
}

go func() {
    buf := make([]byte, 1024)
    for {
        runner.bufMu.Lock()
        n, err = runner.bufOut.Read(buf)
        runner.bufMu.Unlock()
        // ... rest of the code
    }
}()
```

## Specific Errors Found in Logs

### 1. OpenAI Tool Message Sequencing Error

```sh
ERRO <provider/openai.go:160> POST "https://api.openai.com/v1/chat/completions": 400 Bad Request {
  "error": {
    "message": "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.",
    "type": "invalid_request_error",
    "param": "messages.[3].role",
    "code": null
  }
}
```

Root cause in `pkg/provider/openai.go`:

```go
prvdr.params.Messages = append(
    prvdr.params.Messages,
    &Message{
        Role:    MessageRoleTool,
        Content: toolCall.JSON.RawJSON(),  // Adding tool message without preceding tool_calls
    },
)
```

Should be:

```go
prvdr.params.Messages = append(
    prvdr.params.Messages,
    &Message{
        Role:      MessageRoleAssistant,
        Content:   "",
        ToolCalls: toolCalls,  // First add tool_calls
    },
    &Message{
        Role:      MessageRoleTool,
        Content:   toolCall.JSON.RawJSON(),  // Then add tool response
        Reference: toolCall.ID,
    },
)
```

### 2. JSON Parsing Error in Runner

```
19:28:51 ERRO <environment/runner.go:64> unexpected end of JSON input
```

Root cause in `pkg/tools/environment/runner.go`:

```go
command := datura.GetMetaValue[string](artifact, "command")
if command == "" {
    return errnie.Error(errors.New("no command"))
}
// Command executed without proper JSON validation
runner.bufIn.Write([]byte(command))
```

Should include JSON validation:

```go
command := datura.GetMetaValue[string](artifact, "command")
if command == "" {
    return errnie.Error(errors.New("no command"))
}
if !json.Valid([]byte(command)) {
    return errnie.Error(fmt.Errorf("invalid JSON in command: %s", command))
}
runner.bufIn.Write([]byte(command))
```

### 3. Environment Tool Initialization Error

The environment tool fails to initialize properly because the containerd daemon connection isn't properly synchronized. In `cmd/example.go`:

```go
// Current implementation
go func() {
    if err := command.App().RunContext(
        cmd.Context(),
        os.Args,
    ); errnie.Error(err) != nil {
        os.Exit(1)
    }
}()

// Wait for the containerd daemon to start.
for {
    conn, err := client.New(
        "/var/run/containerd/containerd.sock",
        client.WithDefaultNamespace("caramba"),
    )
    if errnie.Error(err) != nil {
        time.Sleep(1 * time.Second)
        continue
    }
    conn.Close()
    break
}
```

Should use proper synchronization:

```go
daemonReady := make(chan struct{})
go func() {
    defer close(daemonReady)
    if err := command.App().RunContext(
        cmd.Context(),
        os.Args,
    ); err != nil {
        errnie.Error(err)
        return
    }
}()

// Wait with timeout
select {
case <-daemonReady:
    // Daemon started successfully
case <-time.After(30 * time.Second):
    return fmt.Errorf("timeout waiting for containerd daemon")
}
```

### 4. Buffer Race Condition

In `pkg/tools/environment/runner.go`, there's a race condition in buffer access:

```go
go func() {
    buf := make([]byte, 1024)
    for {
        n, err = runner.bufOut.Read(buf)  // Concurrent access without synchronization
        if n == 0 || err != nil {
            select {
            case runner.outputReady <- struct{}{}:
            default:
            }
            continue
        }
    }
}()
```

This leads to potential data corruption and the "unexpected end of JSON input" error. Should use mutex:

```go
type Runner struct {
    bufMu sync.Mutex
    // ... other fields
}

go func() {
    buf := make([]byte, 1024)
    for {
        runner.bufMu.Lock()
        n, err = runner.bufOut.Read(buf)
        runner.bufMu.Unlock()
        // ... rest of the code
    }
}()
```

These errors show a pattern of:

1. Improper message sequencing in the OpenAI API integration
2. Lack of proper JSON validation
3. Race conditions in concurrent operations
4. Inadequate synchronization in service initialization

Each of these issues needs to be addressed to improve the stability and reliability of the system.

## OpenAI Tool Calls Implementation Issues

### 1. Provider-Agnostic Message Format Issue

The core issue is that our provider-agnostic `Message` struct in `pkg/provider/types.go` doesn't properly handle tool calls:

```go
// Current implementation
type Message struct {
    ID        string      `json:"id"`
    Reference string      `json:"reference"`
    Role      MessageRole `json:"role"`
    Name      string      `json:"name"`
    Content   string      `json:"content"`
}
```

This leads to mixing OpenAI-specific format with our generic format when handling tool calls:

```go
// Current problematic implementation in handleSingleRequest
prvdr.params.Messages = append(
    prvdr.params.Messages,
    &Message{
        Role:    MessageRoleTool,
        Content: toolCall.JSON.RawJSON(),  // Wrong: Mixing OpenAI format with our generic format
    },
)
```

When we try to reconstruct the messages in `buildMessages`, this causes errors because we're trying to parse OpenAI's format as our generic format:

```go
// Current problematic implementation in buildMessages
case "tool":
    tool := &ToolCall{}
    if err = json.Unmarshal([]byte(message.Content), tool); err != nil {
        return errnie.Error(err)
    }
    messages = append(messages, openai.ToolMessage(tool.ID, message.Content))
```

The solution requires:

1. Update the `Message` struct to properly handle tool calls in our format:

```go
type Message struct {
    ID        string      `json:"id"`
    Reference string      `json:"reference"`
    Role      MessageRole `json:"role"`
    Name      string      `json:"name"`
    Content   string      `json:"content"`
    Tool      *ToolCall   `json:"tool,omitempty"`  // Add proper tool call support
}
```

2. Convert OpenAI's tool calls to our format in `handleSingleRequest`:

```go
// First add the assistant's message
prvdr.params.Messages = append(prvdr.params.Messages, &Message{
    Role:    MessageRoleAssistant,
    Content: completion.Choices[0].Message.Content,
})

// Then add each tool call as a separate message
for _, toolCall := range toolCalls {
    tool := &ToolCall{
        ID:   toolCall.ID,
        Type: toolCall.Type,
        Function: Function{
            Name:      toolCall.Function.Name,
            Arguments: toolCall.Function.Arguments,
        },
    }

    prvdr.params.Messages = append(prvdr.params.Messages, &Message{
        Role:    MessageRoleTool,
        Name:    toolCall.Function.Name,
        Content: toolCall.Function.Arguments,
        Tool:    tool,
    })
}
```

3. Update `buildMessages` to properly handle tool messages:

```go
case "tool":
    if message.Tool == nil {
        return errnie.Error(errors.New("tool message missing tool data"))
    }
    messages = append(messages, openai.ToolMessage(message.Tool.ID, message.Content))
```

This solution:

- Maintains clean separation between provider-agnostic and provider-specific formats
- Properly structures tool call data in our format
- Correctly reconstructs OpenAI's format when needed
- Fixes the "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'" error by ensuring proper message sequencing

The error occurs because we're currently:

1. Not maintaining our provider-agnostic format consistently
2. Mixing OpenAI-specific format into our generic message format
3. Not properly handling the conversion between our format and OpenAI's format

This is a structural issue in how we handle tool calls across different providers, not just a simple message sequencing problem.

## Error Handling and Resource Cleanup Issues

### 1. Container Resource Cleanup

In `pkg/tools/environment/container.go`, container cleanup is incomplete:

```go
// Current implementation - missing cleanup
func (container *Container) Load() (err error) {
    // ... existing code ...

    // Import the image into containerd using OCI format
    images, err := container.conn.Import(
        context.Background(),
        bytes.NewReader(imgBytes),
        client.WithAllPlatforms(true),
        client.WithIndexName(imageName),
    )

    if err != nil {
        return errnie.Error(err)  // No cleanup of temporary resources
    }

    // ... rest of code ...
}
```

Should include proper cleanup:

```go
func (container *Container) Load() (err error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()

    // Create temp directory for build context
    tempDir, err := os.MkdirTemp("", "caramba-build-*")
    if err != nil {
        return errnie.Error(fmt.Errorf("failed to create temp dir: %w", err))
    }
    defer os.RemoveAll(tempDir)

    // Import the image with cleanup on error
    images, err := container.conn.Import(
        ctx,
        bytes.NewReader(imgBytes),
        client.WithAllPlatforms(true),
        client.WithIndexName(imageName),
    )

    if err != nil {
        // Cleanup any partially imported images
        if len(images) > 0 {
            for _, img := range images {
                container.conn.ImageService().Delete(ctx, img.Name)
            }
        }
        return errnie.Error(fmt.Errorf("failed to import image: %w", err))
    }

    // ... rest of code ...
}
```

### 2. Runner Task Management

In `pkg/tools/environment/runner.go`, task management has potential resource leaks:

```go
// Current implementation
func NewRunner(container *Container) *Runner {
    // ... existing code ...

    go func() {
        fifos, err := cio.NewFIFOSetInDir("", container.container.ID(), false)
        if errnie.Error(err) != nil {
            return  // FIFOs might not be cleaned up
        }

        if task, err = container.container.NewTask(ctx, cio.NewCreator()); err != nil {
            errnie.Error(err)
            return  // Task resources might leak
        }

        // ... rest of code ...
    }()

    return runner
}
```

Should include proper error handling and cleanup:

```go
func NewRunner(container *Container) *Runner {
    // ... existing code ...

    errChan := make(chan error, 1)
    go func() {
        defer close(errChan)

        // Create FIFO set with cleanup
        fifos, err := cio.NewFIFOSetInDir("", container.container.ID(), false)
        if err != nil {
            errChan <- fmt.Errorf("failed to create FIFOs: %w", err)
            return
        }
        defer func() {
            if err := fifos.Close(); err != nil {
                errnie.Error(fmt.Errorf("failed to close FIFOs: %w", err))
            }
        }()

        // Create and manage task with proper cleanup
        task, err := container.container.NewTask(ctx, cio.NewCreator())
        if err != nil {
            errChan <- fmt.Errorf("failed to create task: %w", err)
            return
        }

        // Set up task cleanup
        defer func() {
            if err := task.Delete(ctx); err != nil {
                errnie.Error(fmt.Errorf("failed to delete task: %w", err))
            }
        }()

        // ... rest of code ...
    }()

    // Wait for initialization or error
    select {
    case err := <-errChan:
        if err != nil {
            errnie.Error(err)
            return nil
        }
    case <-time.After(30 * time.Second):
        errnie.Error("timeout waiting for runner initialization")
        return nil
    }

    return runner
}
```

### 3. Buffer Resource Management

In `pkg/stream/buffer.go`, buffer cleanup is incomplete:

```go
// Current implementation
func (buffer *Buffer) Close() error {
    errnie.Debug("stream.Buffer.Close")
    return buffer.artifact.Close()
}
```

Should include comprehensive cleanup:

```go
func (buffer *Buffer) Close() error {
    errnie.Debug("stream.Buffer.Close")

    if buffer == nil {
        return nil
    }

    var errs []error

    // Close artifact if it exists
    if buffer.artifact != nil {
        if err := buffer.artifact.Close(); err != nil {
            errs = append(errs, fmt.Errorf("failed to close artifact: %w", err))
        }
    }

    // Reset buffer state
    buffer.artifact = nil
    buffer.fn = nil

    // Return combined errors if any
    if len(errs) > 0 {
        return errnie.Error(fmt.Errorf("multiple close errors: %v", errs))
    }

    return nil
}
```

### 4. File System Operations

In `pkg/fs/store.go`, file operations lack proper error handling:

```go
// Current implementation
func (store *Store) Write(p []byte) (n int, err error) {
    errnie.Debug("fs.Store.Write")
    return store.buffer.Write(p)
}
```

Should include proper validation and error handling:

```go
func (store *Store) Write(p []byte) (n int, err error) {
    errnie.Debug("fs.Store.Write")

    if store == nil {
        return 0, errnie.Error(errors.New("store is nil"))
    }

    if store.buffer == nil {
        return 0, errnie.Error(errors.New("store buffer is nil"))
    }

    if len(p) == 0 {
        return 0, errnie.Error(errors.New("empty write"))
    }

    n, err = store.buffer.Write(p)
    if err != nil {
        return n, errnie.Error(fmt.Errorf("failed to write to buffer: %w", err))
    }

    return n, nil
}
```

These examples demonstrate several critical issues:

1. Incomplete resource cleanup in error cases
2. Missing timeout handling for long-running operations
3. Inadequate error context and wrapping
4. Potential resource leaks in concurrent operations
5. Lack of proper validation before operations
6. Missing cleanup in deferred functions

Each of these issues could lead to resource leaks, hanging operations, or system instability under error conditions.

## Concurrent Operations and Synchronization Issues

### 1. Pump Race Conditions

In `pkg/workflow/pump.go`, the pump implementation has potential race conditions:

```go
// Current implementation with race conditions
type Pump struct {
    pipeline    io.ReadWriteCloser
    buffer      *stream.Buffer
    passthrough *bytes.Buffer  // Shared buffer without synchronization
    done        chan struct{}
    wg          *sync.WaitGroup
}

func NewPump(pipeline io.ReadWriteCloser) {
    passthrough := bytes.NewBuffer([]byte{})  // No mutex protection

    for {
        select {
        case <-done:
            wg.Done()
            return
        default:
            if err = NewFlipFlop(passthrough, pipeline); err != nil {
                errnie.Error(err)
                return
            }
        }
    }
}
```

Should use proper synchronization:

```go
type Pump struct {
    pipeline    io.ReadWriteCloser
    buffer      *stream.Buffer
    passthrough *bytes.Buffer
    bufferMu    sync.RWMutex  // Add mutex for buffer access
    done        chan struct{}
    wg          *sync.WaitGroup
}

func NewPump(pipeline io.ReadWriteCloser) {
    pump := &Pump{
        pipeline:    pipeline,
        passthrough: bytes.NewBuffer([]byte{}),
        bufferMu:    sync.RWMutex{},
        done:        make(chan struct{}),
        wg:          &sync.WaitGroup{},
    }

    pump.wg.Add(1)
    go func() {
        defer pump.wg.Done()
        for {
            select {
            case <-pump.done:
                return
            default:
                pump.bufferMu.Lock()
                err := NewFlipFlop(pump.passthrough, pump.pipeline)
                pump.bufferMu.Unlock()
                if err != nil {
                    errnie.Error(err)
                    return
                }
            }
        }
    }()
}
```

### 2. Store Singleton Pattern Issues

In `pkg/fs/store.go`, the singleton pattern implementation could be improved:

```go
// Current implementation
var once sync.Once
var store *Store

func NewStore() *Store {
    once.Do(func() {
        store = &Store{
            buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
                // ... buffer operations ...
            }),
            conn: conn,
        }
    })
    return store
}
```

Should include proper error handling and thread-safe initialization:

```go
var (
    storeInstance *Store
    storeOnce     sync.Once
    storeMu       sync.RWMutex
)

func NewStore() (*Store, error) {
    var initErr error
    storeOnce.Do(func() {
        conn := NewConn()
        if conn == nil {
            initErr = errors.New("failed to create connection")
            return
        }

        storeInstance = &Store{
            buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
                storeMu.Lock()
                defer storeMu.Unlock()
                // ... buffer operations ...
            }),
            conn: conn,
        }
    })

    if initErr != nil {
        return nil, fmt.Errorf("store initialization failed: %w", initErr)
    }

    return storeInstance, nil
}
```

### 3. Pipeline Concurrent Access

In `pkg/workflow/pipeline.go`, the pipeline lacks proper synchronization:

```go
// Current implementation
type Pipeline struct {
    components []io.ReadWriter
    processed  bool  // Shared flag without protection
}

func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
    if !pipeline.processed {
        for i := range len(pipeline.components) - 1 {
            nn, err = io.Copy(pipeline.components[i+1], pipeline.components[i])
            // ... processing ...
        }
        pipeline.processed = true  // Race condition possible here
    }
    // ... rest of the code ...
}
```

Should use proper synchronization:

```go
type Pipeline struct {
    components []io.ReadWriter
    processed  bool
    mu         sync.RWMutex
}

func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
    pipeline.mu.Lock()
    isProcessed := pipeline.processed
    pipeline.mu.Unlock()

    if !isProcessed {
        pipeline.mu.Lock()
        for i := range len(pipeline.components) - 1 {
            nn, err = io.Copy(pipeline.components[i+1], pipeline.components[i])
            // ... processing ...
        }
        pipeline.processed = true
        pipeline.mu.Unlock()
    }
    // ... rest of the code ...
}
```

### 4. Buffer Concurrent Access

In `pkg/stream/buffer.go`, the buffer implementation needs synchronization:

```go
// Current implementation
type Buffer struct {
    artifact *datura.Artifact  // Shared without protection
    fn       func(*datura.Artifact) error
}

func (buffer *Buffer) Write(p []byte) (n int, err error) {
    if buffer.artifact == nil {
        return 0, io.EOF
    }
    return buffer.artifact.Write(p)  // Concurrent access possible
}
```

Should include proper synchronization:

```go
type Buffer struct {
    artifact *datura.Artifact
    fn       func(*datura.Artifact) error
    mu       sync.RWMutex
}

func (buffer *Buffer) Write(p []byte) (n int, err error) {
    buffer.mu.Lock()
    defer buffer.mu.Unlock()

    if buffer.artifact == nil {
        return 0, io.EOF
    }

    return buffer.artifact.Write(p)
}

func (buffer *Buffer) Read(p []byte) (n int, err error) {
    buffer.mu.RLock()
    defer buffer.mu.RUnlock()

    if buffer.artifact == nil {
        return 0, io.EOF
    }

    return buffer.artifact.Read(p)
}
```

These examples demonstrate several critical synchronization issues:

1. Lack of mutex protection for shared resources
2. Race conditions in singleton pattern implementations
3. Unprotected flag variables in concurrent operations
4. Missing synchronization in buffer operations
5. Improper handling of concurrent initialization
6. Potential deadlocks in nested lock acquisitions

Each of these issues could lead to data races, inconsistent state, or application crashes under concurrent load.
