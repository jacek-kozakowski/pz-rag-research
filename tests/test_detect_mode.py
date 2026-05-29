import pytest
from unittest.mock import patch, MagicMock

from agents.nodes.detect_mode import detect_mode, _keyword_hint, _PROJECT_KEYWORDS, _LEARNING_KEYWORDS


def _mock_llm(reply: str) -> MagicMock:
    response = MagicMock()
    response.content = reply
    llm = MagicMock()
    llm.invoke.return_value = response
    return llm




class TestKeywordHint:
    def test_returns_project_for_build(self):
        assert _keyword_hint("build a REST API") == "project"

    def test_returns_project_for_create(self):
        assert _keyword_hint("create a habit tracker app") == "project"

    def test_returns_project_for_implement(self):
        assert _keyword_hint("implement a queue in Python") == "project"

    def test_returns_learning_for_explain(self):
        assert _keyword_hint("explain quicksort") == "learning"

    def test_returns_learning_for_what_is(self):
        assert _keyword_hint("what is a B-tree?") == "learning"

    def test_returns_learning_for_polish_notatki(self):
        assert _keyword_hint("notatki z algorytmów") == "learning"

    def test_returns_learning_for_polish_wytlumacz(self):
        assert _keyword_hint("wytłumacz jak działa TCP") == "learning"

    def test_returns_none_for_ambiguous(self):
        assert _keyword_hint("Python programming") is None

    def test_case_insensitive(self):
        assert _keyword_hint("BUILD a todo app") == "project"
        assert _keyword_hint("EXPLAIN sorting") == "learning"

    def test_returns_none_for_empty(self):
        assert _keyword_hint("") is None


# ---------------------------------------------------------------------------
# detect_mode — LLM returns "project"
# ---------------------------------------------------------------------------

class TestDetectModeProject:
    @patch("agents.nodes.detect_mode.get_llm")
    def test_returns_project_when_llm_says_project(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("project")
        assert detect_mode("build a FastAPI backend") == "project"

    @patch("agents.nodes.detect_mode.get_llm")
    def test_returns_project_when_llm_says_project_with_whitespace(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("  project  ")
        assert detect_mode("create a React app") == "project"

    @patch("agents.nodes.detect_mode.get_llm")
    def test_returns_project_when_llm_says_project_uppercase(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("PROJECT")
        assert detect_mode("implement a CLI tool") == "project"

    @patch("agents.nodes.detect_mode.get_llm")
    def test_project_for_fullstack_query(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("project")
        assert detect_mode("set up a FastAPI backend with a React frontend and PostgreSQL") == "project"

    @patch("agents.nodes.detect_mode.get_llm")
    def test_project_for_polish_build_query(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("project")
        assert detect_mode("zbuduj aplikację do śledzenia nawyków") == "project"



class TestDetectModeLearning:
    @patch("agents.nodes.detect_mode.get_llm")
    def test_returns_learning_when_llm_says_learning(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("learning")
        assert detect_mode("explain how quicksort works") == "learning"

    @patch("agents.nodes.detect_mode.get_llm")
    def test_returns_learning_for_polish_notes_query(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("learning")
        assert detect_mode("notatki z baz danych") == "learning"

    @patch("agents.nodes.detect_mode.get_llm")
    def test_returns_learning_for_what_is_question(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("learning")
        assert detect_mode("what is a B-tree and how does it differ from a BST?") == "learning"

    @patch("agents.nodes.detect_mode.get_llm")
    def test_returns_learning_for_concept_question(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("learning")
        assert detect_mode("how does backpropagation work in neural networks") == "learning"

    @patch("agents.nodes.detect_mode.get_llm")
    def test_returns_learning_when_llm_returns_unknown_word(self, mock_get_llm):
        # any non-"project" response falls back to "learning"
        mock_get_llm.return_value = _mock_llm("I cannot determine")
        assert detect_mode("something ambiguous") == "learning"

    @patch("agents.nodes.detect_mode.get_llm")
    def test_returns_learning_when_llm_returns_empty(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("")
        assert detect_mode("some query") == "learning"


# ---------------------------------------------------------------------------
# detect_mode — LLM is authoritative even when keyword hint disagrees
# ---------------------------------------------------------------------------

class TestDetectModeLLMAuthority:
    @patch("agents.nodes.detect_mode.get_llm")
    def test_llm_overrides_keyword_project_hint(self, mock_get_llm):
        # "build" keyword hints project, but LLM says learning
        mock_get_llm.return_value = _mock_llm("learning")
        assert detect_mode("build on what you know about algorithms") == "learning"

    @patch("agents.nodes.detect_mode.get_llm")
    def test_llm_overrides_keyword_learning_hint(self, mock_get_llm):
        # "explain" keyword hints learning, but LLM says project
        mock_get_llm.return_value = _mock_llm("project")
        assert detect_mode("I want to explain my project idea") == "project"



class TestDetectModeLLMSetup:
    @patch("agents.nodes.detect_mode.get_llm")
    def test_calls_get_llm_with_query_planner_task(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("learning")
        detect_mode("some query")
        mock_get_llm.assert_called_once_with(task="query_planner")

    @patch("agents.nodes.detect_mode.get_llm")
    def test_passes_query_to_llm(self, mock_get_llm):
        llm = _mock_llm("learning")
        mock_get_llm.return_value = llm
        detect_mode("my specific query text")
        call_args = llm.invoke.call_args[0][0]
        human_message = call_args[-1]
        assert human_message.content == "my specific query text"