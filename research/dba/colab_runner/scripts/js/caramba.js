(() => {
    const mcpServer = colab.global.notebook.colabMcpServer;

    const addTool = mcpServer._registeredTools.add_code_cell;
    const deleteTool = mcpServer._registeredTools.delete_cell;
    const runTool = mcpServer._registeredTools.run_code_cell;

    const bc = new BroadcastChannel("caramba");

    bc.onmessage = (event) => {
        switch(event.data.type) {
            case "add-cell":
                // Use the MCP tool to add a code cell
                addTool.callback({ code: event.data.cell.code });
                break;
            case "delete-cell":
                // Use the MCP tool to delete a cell
                deleteTool.callback({ cellId: event.data.cell.id });
                break;
            case "run-cell":
                // Use the MCP tool to run a code cell
                runTool.callback({ cellId: event.data.cell.id });
                break;
            default:
                console.log("caramba: unknown message", event.data);
                break;
        }
    };

    console.log("caramba.js loaded - listening for messages");
})()
