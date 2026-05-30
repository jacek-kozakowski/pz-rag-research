import pytest
from unittest.mock import patch, MagicMock

from agents.nodes.scaffolding import scaffolding_node, _detect_stack, _STACK_FALLBACK


BASE_STATE = {
    "query": "build a FastAPI REST API with PostgreSQL",
    "messages": [],
    "local_result": {},
    "web_result": {},
    "summary": "A REST API project using FastAPI and PostgreSQL.",
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


class TestDetectStack:
    @patch("agents.nodes.scaffolding.get_llm")
    def test_returns_stack_dict(self, mock_get_llm):
        chain = MagicMock()
        chain.invoke.return_value = {
            "primary_language": "Python",
            "stack": ["FastAPI", "PostgreSQL"],
            "is_fullstack": False,
        }
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.scaffolding.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain)))
            result = _detect_stack("build a FastAPI app")

        assert result["primary_language"] == "Python"
        assert "FastAPI" in result["stack"]

    @patch("agents.nodes.scaffolding.get_llm")
    def test_returns_fallback_on_exception(self, mock_get_llm):
        chain = MagicMock()
        chain.invoke.side_effect = Exception("LLM error")
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.scaffolding.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain)))
            result = _detect_stack("build something")

        assert result == _STACK_FALLBACK


class TestScaffoldingNode:
    @patch("agents.nodes.scaffolding._detect_stack")
    @patch("agents.nodes.scaffolding.get_llm")
    def test_returns_scaffold_and_language(self, mock_get_llm, mock_detect):
        mock_detect.return_value = {"primary_language": "Python", "stack": ["FastAPI"], "is_fullstack": False}
        chain = MagicMock()
        chain.invoke.return_value = [
            {"filepath": "main.py", "purpose": "Entry point", "code": "# main"},
            {"filepath": "models.py", "purpose": "DB models", "code": "# models"},
        ]
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.scaffolding.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain)))
            result = scaffolding_node(_state())

        assert "scaffold" in result
        assert "language" in result
        assert result["language"] == "Python"
        assert len(result["scaffold"]) == 2

    @patch("agents.nodes.scaffolding._detect_stack")
    def test_returns_empty_scaffold_when_no_summary(self, mock_detect):
        result = scaffolding_node(_state(summary=""))
        assert result == {"scaffold": [], "language": "Python"}
        mock_detect.assert_not_called()

    @patch("agents.nodes.scaffolding._detect_stack")
    @patch("agents.nodes.scaffolding.get_llm")
    def test_returns_empty_scaffold_on_llm_json_error(self, mock_get_llm, mock_detect):
        mock_detect.return_value = {"primary_language": "Python", "stack": ["FastAPI"], "is_fullstack": False}
        chain = MagicMock()
        chain.invoke.side_effect = Exception("JSON parse error")
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.scaffolding.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain)))
            result = scaffolding_node(_state())

        assert result["scaffold"] == []

    @patch("agents.nodes.scaffolding._detect_stack")
    @patch("agents.nodes.scaffolding.get_llm")
    def test_returns_empty_scaffold_when_llm_returns_non_list(self, mock_get_llm, mock_detect):
        mock_detect.return_value = {"primary_language": "Go", "stack": ["Go"], "is_fullstack": False}
        chain = MagicMock()
        chain.invoke.return_value = {"filepath": "main.go", "purpose": "entry"}
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.scaffolding.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain)))
            result = scaffolding_node(_state(query="build a Go API"))

        assert result["scaffold"] == []

    @patch("agents.nodes.scaffolding._detect_stack")
    @patch("agents.nodes.scaffolding.get_llm")
    def test_language_matches_detected_stack(self, mock_get_llm, mock_detect):
        mock_detect.return_value = {"primary_language": "TypeScript", "stack": ["React", "Node.js"], "is_fullstack": True}
        chain = MagicMock()
        chain.invoke.return_value = [{"filepath": "index.ts", "purpose": "Entry", "code": "// ts"}]
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.scaffolding.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain)))
            result = scaffolding_node(_state(query="build a React app"))

        assert result["language"] == "TypeScript"

    @patch("agents.nodes.scaffolding._detect_stack")
    @patch("agents.nodes.scaffolding.get_llm")
    def test_passes_query_and_summary_to_scaffold_chain(self, mock_get_llm, mock_detect):
        mock_detect.return_value = {"primary_language": "Python", "stack": ["Flask"], "is_fullstack": False}
        chain = MagicMock()
        chain.invoke.return_value = []
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.scaffolding.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain)))
            scaffolding_node(_state(query="build a Flask app", summary="Flask-based REST API"))

        invoke_args = chain.invoke.call_args[0][0]
        assert "Flask app" in invoke_args.get("query", "")
        assert "Flask-based REST API" in invoke_args.get("summary", "")