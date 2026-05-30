import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from agents.nodes.research import (
    research_agent_node,
    research_tools_node_handler,
    should_continue_research,
)


BASE_STATE = {
    "query": "explain neural networks",
    "messages": [],
    "local_result": {},
    "web_result": {},
    "summary": "",
    "tasks": [],
    "mode": "learning",
    "create_repo": False,
    "use_calendar": False,
    "github_issues": [],
    "readme": "",
    "notes": "",
    "calendar_events": [],
    "intent": "research",
    "scaffold": [],
    "language": "Python",
    "web_enabled": True,
}


def _state(**kwargs):
    return {**BASE_STATE, **kwargs}


def _ai_message_no_tools(content="Here is my answer"):
    msg = MagicMock(spec=AIMessage)
    msg.content = content
    msg.tool_calls = []
    return msg


def _ai_message_with_tools(tool_calls):
    msg = MagicMock(spec=AIMessage)
    msg.content = ""
    msg.tool_calls = tool_calls
    return msg


class TestResearchAgentNode:
    @patch("agents.nodes.research.get_llm")
    def test_returns_messages_update(self, mock_get_llm):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.return_value = _ai_message_no_tools()
        mock_get_llm.return_value = llm

        result = research_agent_node(_state())

        assert "messages" in result
        assert len(result["messages"]) == 1

    @patch("agents.nodes.research.get_llm")
    def test_web_enabled_includes_web_tool(self, mock_get_llm):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.return_value = _ai_message_no_tools()
        mock_get_llm.return_value = llm

        research_agent_node(_state(web_enabled=True))

        bound_tools = llm.bind_tools.call_args.kwargs.get("tools") or llm.bind_tools.call_args[1].get("tools") or llm.bind_tools.call_args[0][0]
        tool_names = [t.name for t in bound_tools]
        assert "search_web_tool" in tool_names

    @patch("agents.nodes.research.get_llm")
    def test_web_disabled_excludes_web_tool(self, mock_get_llm):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.return_value = _ai_message_no_tools()
        mock_get_llm.return_value = llm

        research_agent_node(_state(web_enabled=False))

        bound_tools = llm.bind_tools.call_args.kwargs.get("tools") or llm.bind_tools.call_args[1].get("tools") or llm.bind_tools.call_args[0][0]
        tool_names = [t.name for t in bound_tools]
        assert "search_web_tool" not in tool_names

    @patch("agents.nodes.research.get_llm")
    def test_prepends_system_message(self, mock_get_llm):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.return_value = _ai_message_no_tools()
        mock_get_llm.return_value = llm

        research_agent_node(_state())

        messages_sent = llm.invoke.call_args[0][0]
        from langchain_core.messages import SystemMessage
        assert isinstance(messages_sent[0], SystemMessage)


class TestResearchToolsNodeHandler:
    @patch("agents.nodes.research.decompose_topic_tool")
    def test_handles_decompose_topic_tool(self, mock_decompose):
        mock_decompose.invoke.return_value = ["subtopic1", "subtopic2"]
        mock_decompose.name = "decompose_topic_tool"  # set attribute for clarity

        tool_call = {"name": "decompose_topic_tool", "args": {"topic": "ML"}, "id": "call_1"}
        msg = _ai_message_with_tools([tool_call])

        state = _state(messages=[msg])
        result = research_tools_node_handler(state)

        assert "messages" in result
        assert any(isinstance(m, ToolMessage) for m in result["messages"])

    @patch("agents.nodes.research.search_local_documents_tool")
    def test_handles_search_local_documents_tool(self, mock_local):
        mock_local.invoke.return_value = {"answer": "local answer", "sources": []}
        mock_local.name = "search_local_documents_tool"

        tool_call = {"name": "search_local_documents_tool", "args": {"topics": ["ML"]}, "id": "call_2"}
        msg = _ai_message_with_tools([tool_call])

        state = _state(messages=[msg])
        result = research_tools_node_handler(state)

        assert result["local_result"] == {"answer": "local answer", "sources": []}

    @patch("agents.nodes.research.search_web_tool")
    def test_handles_search_web_tool_when_enabled(self, mock_web):
        mock_web.invoke.return_value = {"answer": "web answer"}
        mock_web.name = "search_web_tool"

        tool_call = {"name": "search_web_tool", "args": {"query": "ML"}, "id": "call_3"}
        msg = _ai_message_with_tools([tool_call])

        state = _state(messages=[msg], web_enabled=True)
        result = research_tools_node_handler(state)

        assert result["web_result"] == {"answer": "web answer"}

    @patch("agents.nodes.research.search_web_tool")
    def test_web_tool_disabled_returns_disabled_message(self, mock_web):
        mock_web.name = "search_web_tool"

        tool_call = {"name": "search_web_tool", "args": {"query": "ML"}, "id": "call_4"}
        msg = _ai_message_with_tools([tool_call])

        state = _state(messages=[msg], web_enabled=False)
        result = research_tools_node_handler(state)

        mock_web.invoke.assert_not_called()
        tool_msg = result["messages"][0]
        assert "disabled" in tool_msg.content.lower()

    def test_unknown_tool_returns_error_message(self):
        tool_call = {"name": "unknown_tool_xyz", "args": {}, "id": "call_5"}
        msg = _ai_message_with_tools([tool_call])

        state = _state(messages=[msg])
        result = research_tools_node_handler(state)

        tool_msg = result["messages"][0]
        assert "unknown" in tool_msg.content.lower() or "not found" in tool_msg.content.lower()

    def test_returns_tool_messages_list(self):
        tool_calls = [
            {"name": "unknown_tool_a", "args": {}, "id": "c1"},
            {"name": "unknown_tool_b", "args": {}, "id": "c2"},
        ]
        msg = _ai_message_with_tools(tool_calls)
        state = _state(messages=[msg])
        result = research_tools_node_handler(state)

        assert len(result["messages"]) == 2


class TestShouldContinueResearch:
    def test_returns_tools_when_last_message_has_tool_calls(self):
        msg = _ai_message_with_tools([{"name": "search_web_tool", "args": {}, "id": "c1"}])
        state = _state(messages=[msg])
        assert should_continue_research(state) == "tools"

    def test_returns_summarization_when_no_tool_calls(self):
        msg = _ai_message_no_tools("Final answer")
        state = _state(messages=[msg])
        assert should_continue_research(state) == "summarization"

    def test_returns_summarization_when_tool_calls_empty_list(self):
        msg = _ai_message_with_tools([])
        state = _state(messages=[msg])
        assert should_continue_research(state) == "summarization"