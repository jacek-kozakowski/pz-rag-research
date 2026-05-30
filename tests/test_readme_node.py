import pytest
from unittest.mock import patch, MagicMock

from agents.nodes.readme import readme_node


BASE_STATE = {
    "query": "build a REST API",
    "messages": [],
    "local_result": {},
    "web_result": {},
    "summary": "A summary about REST APIs",
    "tasks": [],
    "mode": "project",
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


def _mock_llm(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    llm = MagicMock()
    llm.invoke.return_value = response
    # LangChain may call the LLM as a RunnableLambda (callable), not via .invoke()
    llm.return_value = response
    return llm


class TestReadmeNode:
    @patch("agents.nodes.readme.get_llm")
    def test_returns_readme_key(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("# My Project\n\nThis is the README.")
        result = readme_node(_state())
        assert "readme" in result
        assert "# My Project" in result["readme"]

    @patch("agents.nodes.readme.get_llm")
    def test_readme_contains_llm_output(self, mock_get_llm):
        expected = "# REST API Project\n\nSetup instructions here."
        mock_get_llm.return_value = _mock_llm(expected)
        result = readme_node(_state())
        assert result["readme"] == expected

    @patch("agents.nodes.readme.get_llm")
    def test_includes_tasks_in_prompt(self, mock_get_llm):
        llm = _mock_llm("# README")
        mock_get_llm.return_value = llm
        tasks = [
            {"title": "Set up DB", "priority": "high", "duration_minutes": 60, "description": "Configure PostgreSQL"},
            {"title": "Write tests", "priority": "medium", "duration_minutes": 30, "description": "Add pytest tests"},
        ]
        readme_node(_state(tasks=tasks))
        # LangChain calls the LLM as a callable (RunnableLambda), so check call_args
        call_str = str(llm.call_args)
        assert "Set up DB" in call_str
        assert "Write tests" in call_str

    @patch("agents.nodes.readme.get_llm")
    def test_works_with_empty_tasks(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("# Project README")
        result = readme_node(_state(tasks=[]))
        assert "readme" in result

    @patch("agents.nodes.readme.get_llm")
    def test_works_with_empty_summary(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("# README")
        result = readme_node(_state(summary=""))
        assert "readme" in result

    @patch("agents.nodes.readme.get_llm")
    def test_passes_query_in_prompt(self, mock_get_llm):
        llm = _mock_llm("# README")
        mock_get_llm.return_value = llm
        readme_node(_state(query="build a habit tracker"))
        call_str = str(llm.call_args)
        assert "habit tracker" in call_str

    @patch("agents.nodes.readme.get_llm")
    def test_task_formatting_includes_priority(self, mock_get_llm):
        llm = _mock_llm("# README")
        mock_get_llm.return_value = llm
        tasks = [{"title": "Deploy", "priority": "high", "duration_minutes": 120, "description": "Deploy to AWS"}]
        readme_node(_state(tasks=tasks))
        call_str = str(llm.call_args)
        assert "high" in call_str

    @patch("agents.nodes.readme.get_llm")
    def test_returns_only_readme_key_in_dict(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("content")
        result = readme_node(_state())
        assert list(result.keys()) == ["readme"]