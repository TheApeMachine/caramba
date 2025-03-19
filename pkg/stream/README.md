# Stream 🌊

The stream package provides a sophisticated buffering and streaming system built on top of Go's io interfaces. It offers bidirectional streaming with built-in support for the Datura artifact system, making it perfect for handling complex data flows with rich context.

## Why Stream? 🤔

Modern applications often need to handle continuous data flows while maintaining context and type safety. Traditional buffering solutions can be limiting and often don't preserve data context. The stream package solves this by:

- Providing bidirectional streaming with context preservation
- Integrating seamlessly with the Datura artifact system
- Offering flexible data transformation capabilities
- Maintaining type safety throughout the streaming process

## Features ✨

- **Bidirectional Streaming**: Full duplex communication support
- **Artifact Integration**: Native support for Datura artifacts
- **Flexible Processing**: Custom handler functions for stream processing
- **Type Safety**: Maintain data types throughout the stream
- **Error Handling**: Comprehensive error management
- **Resource Management**: Automatic cleanup of resources
- **Performance Optimized**: Efficient memory usage and processing

## Quick Start 🚀

```go
import (
    "github.com/theapemachine/caramba/pkg/stream"
    "github.com/theapemachine/caramba/pkg/datura"
)

// Create a new buffer with a custom handler
buffer := stream.NewBuffer(func(artifact *datura.Artifact) error {
    // Process the artifact
    return nil
})

// Write data to the buffer
data := []byte("Hello, Stream!")
n, err := buffer.Write(data)

// Read processed data
output := make([]byte, 1024)
n, err = buffer.Read(output)
```

## Core Components 🛠️

### Buffer

The `Buffer` type is the main component, providing:

- Bidirectional streaming capabilities
- Integration with Datura artifacts
- Custom processing functions
- Automatic resource management

### Handler Functions 🎯

Handler functions allow you to:

- Process data as it flows through the buffer
- Transform artifacts in-stream
- Implement custom business logic
- Filter and validate data

## Advanced Usage 💡

### Custom Processing Pipeline

```go
// Create a processing pipeline
buffer := stream.NewBuffer(func(artifact *datura.Artifact) error {
    // Validate the artifact
    if err := validate(artifact); err != nil {
        return err
    }

    // Transform the data
    if err := transform(artifact); err != nil {
        return err
    }

    // Enrich with metadata
    return enrich(artifact)
})
```

### Integration with Workflow

```go
// Create a streaming workflow
pipeline := workflow.NewPipeline(
    stream.NewBuffer(validateFn),
    stream.NewBuffer(transformFn),
    stream.NewBuffer(enrichFn),
)
```

## Performance ⚡

The stream package is optimized for:

- Minimal memory allocation
- Efficient data copying
- Resource reuse
- Automatic cleanup

## Integration with Other Packages 🔌

The stream package works seamlessly with:

- `workflow`: For creating complex processing pipelines
- `datura`: For secure artifact handling
- `errnie`: For error management

## Best Practices 📚

1. **Resource Management**

   - Always close buffers when done
   - Use appropriate buffer sizes
   - Handle errors properly

2. **Handler Functions**

   - Keep them focused and simple
   - Return meaningful errors
   - Avoid blocking operations

3. **Performance**
   - Reuse buffers when possible
   - Process data in chunks
   - Implement backpressure when needed
