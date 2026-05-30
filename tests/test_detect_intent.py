import pytest
from unittest.mock import patch, MagicMock

from agents.nodes.detect_intent import detect_intent_node, route_by_intent


def _mock_llm(reply: str) -> MagicMock:
    response = MagicMock()
    response.content = reply
    llm = MagicMock()
    llm.invoke.return_value = response
    return llm


BASE_STATE = {
    "query": "explain quicksort",
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
    "web_enabled": False,
}


def _state(**kwargs):
    return {**BASE_STATE, **kwargs}


class TestDetectIntentNode:
    @patch("agents.nodes.detect_intent.get_llm")
    def test_returns_research_when_llm_says_research(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("research")
        result = detect_intent_node(_state(query="how does quicksort work"))
        assert result == {"intent": "research"}

    @patch("agents.nodes.detect_intent.get_llm")
    def test_returns_local_files_when_llm_says_local_files(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("local_files")
        result = detect_intent_node(_state(query="notatki z algorytmów"))
        assert result == {"intent": "local_files"}

    @patch("agents.nodes.detect_intent.get_llm")
    def test_defaults_to_research_for_ambiguous_reply(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("I cannot determine")
        result = detect_intent_node(_state(query="something vague"))
        assert result == {"intent": "research"}

    @patch("agents.nodes.detect_intent.get_llm")
    def test_defaults_to_research_for_empty_reply(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("")
        result = detect_intent_node(_state(query="something"))
        assert result == {"intent": "research"}

    @patch("agents.nodes.detect_intent.get_llm")
    def test_local_files_detected_when_substring_present(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("local_files please")
        result = detect_intent_node(_state(query="make notes from my OS lectures"))
        assert result == {"intent": "local_files"}

    @patch("agents.nodes.detect_intent.get_llm")
    def test_calls_query_planner_task(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("research")
        detect_intent_node(_state())
        mock_get_llm.assert_called_once_with(task="query_planner")

    @patch("agents.nodes.detect_intent.get_llm")
    def test_passes_query_to_llm(self, mock_get_llm):
        llm = _mock_llm("research")
        mock_get_llm.return_value = llm
        detect_intent_node(_state(query="my unique query"))
        messages = llm.invoke.call_args[0][0]
        assert messages[-1].content == "my unique query"


class TestRouteByIntent:
    def test_returns_research_by_default(self):
        assert route_by_intent(_state(intent="research")) == "research"

    def test_returns_local_files(self):
        assert route_by_intent(_state(intent="local_files")) == "local_files"

    def test_returns_research_when_intent_missing(self):
        state = dict(BASE_STATE)
        state.pop("intent", None)
        assert route_by_intent(state) == "research"