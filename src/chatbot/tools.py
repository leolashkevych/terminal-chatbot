import logging
from duckduckgo_search import DDGS
from datetime import datetime
from typing import List, Dict, Any, Callable


class Tools:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.register_tool(
            name="get_current_date",
            func=self.get_current_date,
            description="Returns the current date and time",
            parameters=[],
        )

        self.register_tool(
            name="duckduckgo_search",
            func=self.duckduckgo_search,
            description="Query DuckDuckGo to find up to date information",
            parameters=[
                {"name": "query", "type": "str", "description": "Query to send"}
            ],
        )

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: List[Dict[str, str]],
    ):
        self._tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters,
        }

    def get_tools_description(self) -> str:
        tools_desc = "Available tools:\n\n"
        for name, tool in self._tools.items():
            tools_desc += f"Tool: {name}\n"
            tools_desc += f"Description: {tool['description']}\n"
            if tool["parameters"]:
                tools_desc += "Parameters:\n"
                for param in tool["parameters"]:
                    tools_desc += f"  - {param['name']} ({param['type']}): {param['description']}\n"
            tools_desc += "\n"
        return tools_desc

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        logging.info(f"Executing tool: {tool_name} with parameters: {kwargs}")
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        return self._tools[tool_name]["function"](**kwargs)

    # Tool implementations below
    @staticmethod
    def get_current_date() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def duckduckgo_search(query: str) -> str:
        return DDGS().text(query, max_results=3)
