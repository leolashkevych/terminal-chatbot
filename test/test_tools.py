from chatbot.tools import Tools
import pytest


class TestTools:
    def test_execute_tool_duckduckgo_search(self):
        tools = Tools()
        query = "skeeyee"
        result = tools.execute_tool("duckduckgo_search", query=query)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_unknown_tool(self):
        tools = Tools()
        with pytest.raises(ValueError) as excinfo:
            tools.execute_tool("unknown_tool")
        assert str(excinfo.value) == "Tool 'unknown_tool' not found"
