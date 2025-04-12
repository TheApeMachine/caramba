# Examples

You will find a number of examples here, all of which are single-file implementations, serving as "recipe" style tutorials.

## Terminal Chat Agent Example

This is a simple example of a terminal-based agentic chat using the Caramba framework. It demonstrates how to create a basic interactive agent that can process user input and respond through the terminal.

### Features

- Simple terminal-based interface
- Integration with the Caramba MCP service
- Basic input processing and response generation
- Easy to extend with more advanced capabilities

### How to Run

1. Make sure you have Go installed and the Caramba framework set up
2. Navigate to the project root directory
3. Run the example:

```bash
go run examples/terminal_chat/main.go
```

### Usage

- Type your message and press Enter to send it to the agent
- Type 'exit' to quit the application

### Extending the Example

This is a basic example that can be extended in several ways:

1. Implement more sophisticated input processing in the `processInput` method
2. Add support for different types of commands
3. Integrate with external services or APIs
4. Add support for conversation history
5. Implement more advanced natural language processing

### Implementation Details

The example uses the following components from the Caramba framework:

- `service.MCP`: The Mission Control Protocol service for agent communication
- `tools`: Various tools that can be used by the agent

The main components of the example are:

- `ChatAgent`: A simple agent that processes user input and generates responses
- `StartChatLoop`: The main loop that handles user input and displays responses
- `processInput`: A method that processes user input and generates a response
