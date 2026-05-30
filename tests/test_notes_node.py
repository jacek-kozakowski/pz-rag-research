import pytest
from unittest.mock import patch, MagicMock

from agents.nodes.notes import _fix_latex, notes_node


BASE_STATE = {
    "query": "explain quicksort",
    "messages": [],
    "local_result": {},
    "web_result": {},
    "summary": "A summary about sorting",
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


class TestFixLatex:
    def test_converts_paren_delimiters_to_inline(self):
        result = _fix_latex(r"\(O(n \log n)\)")
        assert result == r"$O(n \log n)$"

    def test_converts_bracket_delimiters_to_display(self):
        result = _fix_latex(r"\[f(x) = x^2\]")
        assert result == r"$$f(x) = x^2$$"

    def test_no_change_when_already_dollar(self):
        text = "$x^2$"
        assert _fix_latex(text) == text

    def test_no_change_on_plain_text(self):
        text = "Hello world"
        assert _fix_latex(text) == text

    def test_multiple_occurrences(self):
        text = r"\(a\) and \(b\)"
        result = _fix_latex(text)
        assert result == "$a$ and $b$"

    def test_wraps_bare_begin_end_env(self):
        text = r"\begin{align}x = 1\end{align}"
        result = _fix_latex(text)
        assert "$$" in result
        assert r"\begin{align}" in result

    def test_does_not_double_wrap_already_wrapped(self):
        text = r"$$\begin{align}x = 1\end{align}$$"
        result = _fix_latex(text)
        assert result.count("$$") == 2


class TestNotesNode:
    @patch("agents.nodes.notes._notes_from_local_files")
    def test_routes_to_local_files_when_intent_is_local_files(self, mock_local):
        mock_local.return_value = {"notes": "local notes"}
        result = notes_node(_state(intent="local_files"))
        mock_local.assert_called_once()
        assert result == {"notes": "local notes"}

    @patch("agents.nodes.notes._notes_from_research")
    def test_routes_to_research_when_intent_is_research(self, mock_research):
        mock_research.return_value = {"notes": "research notes"}
        result = notes_node(_state(intent="research"))
        mock_research.assert_called_once()
        assert result == {"notes": "research notes"}

    @patch("agents.nodes.notes._notes_from_research")
    def test_defaults_to_research_when_no_intent(self, mock_research):
        mock_research.return_value = {"notes": "research notes"}
        state = _state()
        state.pop("intent", None)
        result = notes_node(state)
        mock_research.assert_called_once()

    @patch("agents.nodes.notes._notes_from_local_files")
    def test_local_files_not_called_for_research_intent(self, mock_local):
        with patch("agents.nodes.notes._notes_from_research", return_value={"notes": "x"}):
            notes_node(_state(intent="research"))
        mock_local.assert_not_called()

    @patch("agents.nodes.notes._notes_from_research")
    def test_research_not_called_for_local_files_intent(self, mock_research):
        with patch("agents.nodes.notes._notes_from_local_files", return_value={"notes": "x"}):
            notes_node(_state(intent="local_files"))
        mock_research.assert_not_called()


class TestNotesFromLocalFiles:
    """Tests for _notes_from_local_files — patch at the rag module level since imports are local."""

    def test_returns_no_documents_message_when_no_candidates(self):
        with patch("rag.vector_storage.find_relevant_sources", return_value=[]):
            with patch("agents.nodes.notes._notes_from_local_files") as mock:
                mock.return_value = {"notes": "No relevant documents found for this topic."}
                result = notes_node(_state(intent="local_files"))
        assert "No relevant documents" in result["notes"]

    def test_returns_no_documents_message_when_judge_filters_all(self):
        with patch("agents.nodes.notes._llm_filter_sources", return_value=[]):
            with patch("agents.nodes.notes._notes_from_local_files") as mock:
                mock.return_value = {"notes": "No relevant documents found for this topic."}
                result = notes_node(_state(intent="local_files"))
        assert "No relevant documents" in result["notes"]