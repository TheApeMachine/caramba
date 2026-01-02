from __future__ import annotations

import os
from typing import Any, cast

try:
    # `litellm.responses(..., tools=...)` is typed against OpenAI tool params.
    # LiteLLM's `responses()` expects function tools shaped like:
    #   {"type":"function","name":..., "description":..., "parameters":...}
    from openai.types.responses import ToolParam  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    ToolParam = Any  # type: ignore[misc,assignment]


class FileTool:
    """File tool"""
    def __init__(self) -> None:
        # Tool definitions must include a top-level "name" for LiteLLM/OpenAI
        # function-calling via the Responses API.
        self.read_file_definition = cast(ToolParam, {
            "type": "function",
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
                "additionalProperties": False,
            },
        })

        self.list_files_definition = cast(ToolParam, {
            "type": "function",
            "name": "list_files",
            "description": "List the files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {"directory_path": {"type": "string"}},
                "required": ["directory_path"],
                "additionalProperties": False,
            },
        })

        self.search_text_definition = cast(ToolParam, {
            "type": "function",
            "name": "search_text",
            "description": "Search for text within files under the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_text": {"type": "string"},
                    "file_path": {"type": "string"},
                },
                "required": ["search_text"],
                "additionalProperties": False,
            },
        })

    def handle_tool_call(self, name: str, args: dict[str, Any]) -> str | list[str] | dict[str, Any]:
        """Handle a tool call."""
        if name == "read_file":
            return self.read_file(args["file_path"])
        elif name == "list_files":
            return self.list_files(args["directory_path"])
        elif name == "search_text":
            return self.search_text(args.get("file_path", ""), args["search_text"])
        else:
            return {"error": f"Unknown tool: {name}"}


    def read_file(self, file_path: str) -> str:
        """Read the file"""
        with open(file_path, "r") as f:
            return f.read()

    def list_files(self, directory_path: str) -> list[str]:
        """List the files in the directory"""
        return os.listdir(directory_path)

    def search_text(self, file_path: str, search_text: str) -> list[str]:
        """Search text recursively in files from the current working directory"""
        results: list[str] = []

        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith(".py") or file.endswith(".md") or file.endswith(".txt") or file.endswith(".yml") or file.endswith(".yaml") or file.endswith(".json") or file.endswith(".jsonl") or file.endswith(".h5") or file.endswith(".pt") or file.endswith(".pth") or file.endswith(".pkl") or file.endswith(".pickle") or file.endswith(".csv") or file.endswith(".tsv") or file.endswith(".parquet") or file.endswith(".feather") or file.endswith(".arrow") or file.endswith(".parquet") or file.endswith(".feather") or file.endswith(".arrow") or file.endswith(".parquet") or file.endswith(".feather") or file.endswith(".arrow"):
                    with open(os.path.join(root, file), "r") as f:
                        if search_text in f.read():
                            results.append(os.path.join(root, file))
        return results