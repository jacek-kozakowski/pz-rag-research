import os
from contextlib import ExitStack
from unittest.mock import patch, MagicMock
from pydantic import Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.graph import build_project_graph

FAKE_TOKEN = "ghp_faketoken"

_GET_LLM_TARGETS = [
    "agents.nodes.research.get_llm",
    "research.summarizer.get_llm",
    "research.planner.get_llm",
    "agents.nodes.scaffolding.get_llm",
    "agents.nodes.github_issues.get_llm",
    "agents.nodes.readme.get_llm",
]

# State fields consumed sequentially by each LLM call (in graph execution order).
#
# scaffolding_node makes TWO LLM calls: _detect_stack (returns a dict) + scaffold chain (returns a list).
#
# Without GITHUB_TOKEN (6 responses): research, summary, tasks, detect_stack, scaffold, readme
# With GITHUB_TOKEN + GITHUB_REPO (6 responses): same (no repo slug, no issue generation from LLM)
# With create_repo=True (8 responses): + repo_slug + issues
_R_RESEARCH = "I have gathered enough context to proceed."
_R_SUMMARY = "An AI assistant project that helps users automate workflows."
_R_TASKS = '[{"title":"Set up CI","description":"Configure GitHub Actions","priority":"high","duration_minutes":60,"deadline":"2026-06-01","start_time":"09:00"}]'
_R_DETECT_STACK = '{"primary_language":"Python","stack":["Python"],"is_fullstack":false}'
_R_SCAFFOLD = '[{"filepath":"main.py","purpose":"Entry point","code":"def main(): pass"}]'
_R_REPO_SLUG = "ai-assistant"
_R_ISSUES = '[{"title":"Write tests","description":"Add unit tests for main.py","priority":"high"}]'
_R_README = "# AI Assistant\nA project."

_BASE_STATE = {
    "query": "Build an AI assistant",
    "messages": [],
    "mode": "project",
    "create_repo": False,
    "web_enabled": False,
    "use_calendar": False,
    "tasks": [],
    "scaffold": [],
    "github_issues": [],
    "local_result": {},
    "web_result": {},
    "summary": "",
    "notes": "",
    "readme": "",
    "calendar_events": [],
    "intent": "research",
    "language": "Python",  # pre-set → skips _detect_language LLM call in scaffolding_node
}


class _SequentialFakeLLM(BaseChatModel):
    """Fake LLM that returns responses from a list in order and supports bind_tools."""
    responses: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not self.responses:
            raise ValueError("_SequentialFakeLLM ran out of responses")
        content = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "sequential-fake"


def _mock_response(status, json_data=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    r.text = text
    return r


def _run_graph(state_overrides=None, responses=None, env_overrides=None, extra_patches=None):
    state = {**_BASE_STATE, **(state_overrides or {})}
    fake_llm = _SequentialFakeLLM(responses=list(responses or []))
    env = {**({k: v for k, v in os.environ.items()} if not env_overrides else {}), **(env_overrides or {})}

    with ExitStack() as stack:
        for target in _GET_LLM_TARGETS:
            stack.enter_context(patch(target, return_value=fake_llm))
        for p in (extra_patches or []):
            stack.enter_context(p)
        with patch.dict(os.environ, env, clear=bool(env_overrides is not None and "GITHUB_TOKEN" not in env_overrides)):
            return build_project_graph().invoke(state)


class TestProjectFlowE2E:
    def test_full_flow_produces_github_issues(self):
        responses = [_R_RESEARCH, _R_SUMMARY, _R_TASKS, _R_DETECT_STACK, _R_SCAFFOLD, _R_REPO_SLUG, _R_ISSUES, _R_README]

        mock_get = _mock_response(200, {"login": "testuser"})
        mock_posts = [
            _mock_response(201, {"full_name": "testuser/ai-assistant"}),
            _mock_response(201, {"number": 1, "title": "Write tests",
                                  "html_url": "https://github.com/testuser/ai-assistant/issues/1"}),
            _mock_response(201, {}),  # scaffold comment
        ]
        mock_put = _mock_response(201, {"content": {
            "html_url": "https://github.com/testuser/ai-assistant/blob/main/main.py"
        }})

        result = _run_graph(
            state_overrides={"create_repo": True},
            responses=responses,
            env_overrides={"GITHUB_TOKEN": FAKE_TOKEN},
            extra_patches=[
                patch("agents.nodes.github_issues.requests.get", return_value=mock_get),
                patch("agents.nodes.github_issues.requests.post", side_effect=mock_posts),
                patch("agents.nodes.github_issues.requests.put", return_value=mock_put),
            ],
        )

        issues = result["github_issues"]
        assert len(issues) == 1
        assert issues[0]["number"] == 1
        assert issues[0]["title"] == "Write tests"
        assert issues[0]["repo"] == "testuser/ai-assistant"
        assert issues[0]["url"] == "https://github.com/testuser/ai-assistant/issues/1"

    def test_full_flow_no_token_skips_github(self):
        # Without GITHUB_TOKEN github_issues_node returns early → 6 LLM calls (no issues calls)
        responses = [_R_RESEARCH, _R_SUMMARY, _R_TASKS, _R_DETECT_STACK, _R_SCAFFOLD, _R_README]

        with patch("agents.nodes.github_issues.requests.post") as mock_post:
            result = _run_graph(
                responses=responses,
                env_overrides={},  # clear env → no GITHUB_TOKEN
                extra_patches=[patch("agents.nodes.github_issues.requests.post", new=mock_post)],
            )

        mock_post.assert_not_called()
        assert result["github_issues"] == []

    def test_full_flow_scaffold_populated_by_scaffolding_node(self):
        responses = [_R_RESEARCH, _R_SUMMARY, _R_TASKS, _R_DETECT_STACK, _R_SCAFFOLD, _R_REPO_SLUG, _R_ISSUES, _R_README]

        result = _run_graph(
            state_overrides={"create_repo": True},
            responses=responses,
            env_overrides={"GITHUB_TOKEN": FAKE_TOKEN},
            extra_patches=[
                patch("agents.nodes.github_issues.requests.get",
                      return_value=_mock_response(200, {"login": "u"})),
                patch("agents.nodes.github_issues.requests.post", side_effect=[
                    _mock_response(201, {"full_name": "u/ai-assistant"}),
                    _mock_response(201, {"number": 1, "title": "Write tests",
                                         "html_url": "https://github.com/u/ai-assistant/issues/1"}),
                    _mock_response(201, {}),
                ]),
                patch("agents.nodes.github_issues.requests.put",
                      return_value=_mock_response(201, {"content": {"html_url": "https://github.com/u/ai-assistant/blob/main/main.py"}})),
            ],
        )

        scaffold = result.get("scaffold", [])
        assert len(scaffold) == 1
        assert scaffold[0]["filepath"] == "main.py"
        assert scaffold[0]["purpose"] == "Entry point"

    def test_full_flow_summary_propagated_from_summarization_node(self):
        responses = [_R_RESEARCH, _R_SUMMARY, _R_TASKS, _R_DETECT_STACK, _R_SCAFFOLD, _R_README]

        result = _run_graph(
            responses=responses,
            env_overrides={},
        )

        assert result["summary"] == _R_SUMMARY

    def test_full_flow_readme_populated_by_readme_node(self):
        responses = [_R_RESEARCH, _R_SUMMARY, _R_TASKS, _R_DETECT_STACK, _R_SCAFFOLD, _R_README]

        result = _run_graph(
            responses=responses,
            env_overrides={},
        )

        assert "AI Assistant" in result["readme"]

    def test_full_flow_tasks_used_when_scaffold_empty(self):
        """When scaffold is empty, github_issues_node falls back to state['tasks']."""
        responses = [_R_RESEARCH, _R_SUMMARY, _R_TASKS, _R_DETECT_STACK, "[]", _R_README]

        mock_post = MagicMock(return_value=_mock_response(201, {
            "number": 2, "title": "Set up CI",
            "html_url": "https://github.com/user/repo/issues/2",
        }))

        result = _run_graph(
            responses=responses,
            env_overrides={"GITHUB_TOKEN": FAKE_TOKEN, "GITHUB_REPO": "user/repo"},
            extra_patches=[patch("agents.nodes.github_issues.requests.post", new=mock_post)],
        )

        issues = result["github_issues"]
        assert len(issues) == 1
        assert issues[0]["title"] == "Set up CI"
