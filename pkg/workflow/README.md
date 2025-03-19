# Workflow 🔄

The workflow package provides a powerful and flexible pipeline system for connecting and managing streaming components in Go. It enables seamless data flow between different parts of your application while maintaining the simplicity of Go's `io.Reader` and `io.Writer` interfaces.

## Why Workflow? 🤔

In modern applications, data often needs to flow through multiple processing stages - from input handling to transformation to output generation. Traditional approaches can lead to complex, tightly coupled code. The workflow package solves this by:

- Creating clean, composable data pipelines
- Maintaining loose coupling between components
- Supporting both linear and branching workflows
- Enabling easy testing and modification of data flows

## Features ✨

- **Pipeline Construction**: Easy creation of data processing pipelines
- **Component Chaining**: Connect multiple `io.ReadWriter` components seamlessly
- **Flexible Composition**: Support for nested pipelines and complex workflows
- **Feedback Loops**: Built-in support for workflow feedback mechanisms
- **Type Conversion**: Automatic handling of data type conversions between components
- **Error Handling**: Robust error propagation throughout the pipeline

## Quick Start 🚀

```go
import (
    "github.com/theapemachine/caramba/pkg/workflow"
    "io"
    "os"
)

// Create a simple pipeline
pipeline := workflow.NewPipeline(
    messageComponent,
    processingAgent,
    outputProvider,
)

// Data automatically flows through all components
io.Copy(os.Stdout, pipeline)

// Create nested pipelines for complex workflows
subPipeline := workflow.NewPipeline(validatorComponent, transformerComponent)
mainPipeline := workflow.NewPipeline(
    inputComponent,
    subPipeline,
    outputComponent,
)
```

## Core Components 🛠️

### Pipeline

The `Pipeline` type is the central component, providing:

- Sequential data flow management
- Automatic component connection
- Error propagation
- Resource cleanup

### Feedback System 📢

The feedback system allows components to:

- Send feedback up or down the pipeline
- Modify workflow behavior based on results
- Implement adaptive processing logic

### Type Converter 🔄

The converter system handles:

- Automatic type conversions between components
- Custom conversion logic support
- Safe handling of type mismatches

## Advanced Usage 💡

### Creating Custom Components

```go
type CustomComponent struct {
    // your fields here
}

func (c *CustomComponent) Read(p []byte) (n int, err error) {
    // implement reading logic
}

func (c *CustomComponent) Write(p []byte) (n int, err error) {
    // implement writing logic
}
```

### Nested Pipelines

```go
// Create reusable pipeline components
validation := workflow.NewPipeline(
    schemaValidator,
    typeChecker,
)

transformation := workflow.NewPipeline(
    dataTransformer,
    enricher,
)

// Combine them into a larger workflow
complete := workflow.NewPipeline(
    inputHandler,
    validation,
    transformation,
    outputHandler,
)
```

## Performance ⚡

The workflow package is designed for efficiency:

- Minimal memory allocation
- Zero-copy where possible
- Lazy evaluation of pipeline stages
- Efficient resource cleanup

## Integration with Other Packages 🔌

The workflow package works seamlessly with other caramba packages:

- `stream`: For enhanced buffering and data handling
- `datura`: For secure artifact management
- `errnie`: For comprehensive error handling
