import pytest
from unittest.mock import patch, MagicMock

from agents.nodes.summarization import summarization_node, task_planner_node


BASE_STATE = {
    "query": "explain neural networks",
    "messages": [],
    "local_result": {"answer": "Local data about neural networks"},
    "web_result": {"answer": "Web data about neural networks"},
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


class TestSummarizationNode:
    @patch("agents.nodes.summarization.summarize")
    def test_returns_summary_from_summarize(self, mock_summarize):
        mock_summarize.return_value = {"summary": "This is a summary."}
        result = summarization_node(_state())
        assert result == {"summary": "This is a summary."}

    @patch("agents.nodes.summarization.summarize")
    def test_passes_query_to_summarize(self, mock_summarize):
        mock_summarize.return_value = {"summary": "x"}
        summarization_node(_state(query="my specific query"))
        args = mock_summarize.call_args[0]
        assert args[0] == "my specific query"

    @patch("agents.nodes.summarization.summarize")
    def test_passes_local_result_to_summarize(self, mock_summarize):
        mock_summarize.return_value = {"summary": "x"}
        local = {"answer": "some local content"}
        summarization_node(_state(local_result=local))
        args = mock_summarize.call_args[0]
        assert args[1] == local

    @patch("agents.nodes.summarization.summarize")
    def test_passes_web_result_to_summarize(self, mock_summarize):
        mock_summarize.return_value = {"summary": "x"}
        web = {"answer": "some web content"}
        summarization_node(_state(web_result=web))
        args = mock_summarize.call_args[0]
        assert args[2] == web

    @patch("agents.nodes.summarization.summarize")
    def test_uses_empty_dicts_when_results_missing(self, mock_summarize):
        mock_summarize.return_value = {"summary": "x"}
        state = _state()
        state.pop("local_result", None)
        state.pop("web_result", None)
        summarization_node(state)
        args = mock_summarize.call_args[0]
        assert args[1] == {}
        assert args[2] == {}


class TestTaskPlannerNode:
    @patch("agents.nodes.summarization.plan_task")
    def test_calls_plan_task_for_research_intent(self, mock_plan):
        mock_plan.return_value = [{"title": "Task 1"}]
        result = task_planner_node(_state(intent="research", summary="some summary"))
        mock_plan.assert_called_once_with("some summary", "explain neural networks")
        assert result == {"tasks": [{"title": "Task 1"}]}

    @patch("agents.nodes.summarization.plan_task_from_notes")
    def test_calls_plan_task_from_notes_for_local_files_intent(self, mock_plan_notes):
        mock_plan_notes.return_value = [{"title": "Study task"}]
        result = task_planner_node(_state(intent="local_files", notes="some notes"))
        mock_plan_notes.assert_called_once_with("some notes", "explain neural networks")
        assert result == {"tasks": [{"title": "Study task"}]}

    @patch("agents.nodes.summarization.plan_task")
    def test_passes_empty_summary_when_missing(self, mock_plan):
        mock_plan.return_value = []
        state = _state(intent="research")
        state.pop("summary", None)
        task_planner_node(state)
        args = mock_plan.call_args[0]
        assert args[0] == ""

    @patch("agents.nodes.summarization.plan_task_from_notes")
    def test_passes_empty_notes_when_missing(self, mock_plan_notes):
        mock_plan_notes.return_value = []
        state = _state(intent="local_files")
        state.pop("notes", None)
        task_planner_node(state)
        args = mock_plan_notes.call_args[0]
        assert args[0] == ""

    @patch("agents.nodes.summarization.plan_task")
    def test_returns_tasks_list_in_state(self, mock_plan):
        tasks = [{"title": "A"}, {"title": "B"}]
        mock_plan.return_value = tasks
        result = task_planner_node(_state(intent="research", summary="x"))
        assert result["tasks"] == tasks