import os
import pytest
from unittest.mock import patch, MagicMock

from agents.nodes.github_issues import (
    _get_headers,
    _repo_exists,
    _create_repo,
    _repo_name_from_query,
    github_issues_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_TOKEN = "ghp_faketoken123"


def _mock_response(status_code: int, json_data: dict = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# _get_headers
# ---------------------------------------------------------------------------

class TestGetHeaders:
    def test_returns_correct_auth_header(self):
        headers = _get_headers(FAKE_TOKEN)
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"

    def test_returns_correct_accept_header(self):
        headers = _get_headers(FAKE_TOKEN)
        assert headers["Accept"] == "application/vnd.github+json"

    def test_returns_api_version_header(self):
        headers = _get_headers(FAKE_TOKEN)
        assert headers["X-GitHub-Api-Version"] == "2022-11-28"


# ---------------------------------------------------------------------------
# _repo_exists
# ---------------------------------------------------------------------------

class TestRepoExists:
    @patch("agents.nodes.github_issues.requests.get")
    def test_returns_true_when_200(self, mock_get):
        mock_get.return_value = _mock_response(200, {"full_name": "user/repo"})
        assert _repo_exists(FAKE_TOKEN, "user/repo") is True

    @patch("agents.nodes.github_issues.requests.get")
    def test_returns_false_when_404(self, mock_get):
        mock_get.return_value = _mock_response(404)
        assert _repo_exists(FAKE_TOKEN, "user/repo") is False

    @patch("agents.nodes.github_issues.requests.get")
    def test_calls_correct_endpoint(self, mock_get):
        mock_get.return_value = _mock_response(200)
        _repo_exists(FAKE_TOKEN, "user/my-repo")
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/user/my-repo",
            headers=_get_headers(FAKE_TOKEN),
        )


# ---------------------------------------------------------------------------
# _repo_name_from_query
# ---------------------------------------------------------------------------

class TestRepoNameFromQuery:
    @patch("agents.nodes.github_issues.get_llm")
    def test_returns_slug(self, mock_get_llm):
        llm_response = MagicMock()
        llm_response.content = "my-cool-project"
        chain = MagicMock()
        chain.invoke.return_value = llm_response
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.github_issues.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=chain)
            result = _repo_name_from_query("Build a cool project")

        assert result == "my-cool-project"

    @patch("agents.nodes.github_issues.get_llm")
    def test_strips_invalid_characters(self, mock_get_llm):
        llm_response = MagicMock()
        llm_response.content = "  My_REPO Name!! "
        chain = MagicMock()
        chain.invoke.return_value = llm_response
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.github_issues.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=chain)
            result = _repo_name_from_query("any query")

        assert result == "my-repo-name"

    @patch("agents.nodes.github_issues.get_llm")
    def test_truncates_to_40_chars(self, mock_get_llm):
        llm_response = MagicMock()
        llm_response.content = "a" * 60
        chain = MagicMock()
        chain.invoke.return_value = llm_response
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.github_issues.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=chain)
            result = _repo_name_from_query("any query")

        assert len(result) <= 40

    @patch("agents.nodes.github_issues.get_llm")
    def test_fallback_when_empty_slug(self, mock_get_llm):
        llm_response = MagicMock()
        llm_response.content = "---"
        chain = MagicMock()
        chain.invoke.return_value = llm_response
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("agents.nodes.github_issues.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = MagicMock(return_value=chain)
            result = _repo_name_from_query("any query")

        assert result == "ai-research-project"


# ---------------------------------------------------------------------------
# _create_repo
# ---------------------------------------------------------------------------

class TestCreateRepo:
    def _make_user_resp(self, login="testuser"):
        return _mock_response(200, {"login": login})

    @patch("agents.nodes.github_issues.requests.get")
    @patch("agents.nodes.github_issues.requests.post")
    @patch("agents.nodes.github_issues._repo_name_from_query", return_value="my-repo")
    def test_creates_repo_successfully(self, mock_name, mock_post, mock_get):
        mock_get.return_value = self._make_user_resp()
        mock_post.return_value = _mock_response(201, {"full_name": "testuser/my-repo"})

        result = _create_repo(FAKE_TOKEN, "Build something")

        assert result == "testuser/my-repo"

    @patch("agents.nodes.github_issues.requests.get")
    @patch("agents.nodes.github_issues.requests.post")
    @patch("agents.nodes.github_issues._repo_name_from_query", return_value="existing-repo")
    def test_returns_fullname_on_422_when_repo_exists(self, mock_name, mock_post, mock_get):
        user_resp = self._make_user_resp()
        exists_resp = _mock_response(200)
        mock_get.side_effect = [user_resp, exists_resp]
        mock_post.return_value = _mock_response(422, text="already exists")

        result = _create_repo(FAKE_TOKEN, "some query")

        assert result == "testuser/existing-repo"

    @patch("agents.nodes.github_issues.requests.get")
    @patch("agents.nodes.github_issues.requests.post")
    @patch("agents.nodes.github_issues._repo_name_from_query", return_value="ghost-repo")
    def test_returns_none_on_422_when_repo_not_found(self, mock_name, mock_post, mock_get):
        user_resp = self._make_user_resp()
        not_found_resp = _mock_response(404)
        mock_get.side_effect = [user_resp, not_found_resp]
        mock_post.return_value = _mock_response(422, text="already exists")

        result = _create_repo(FAKE_TOKEN, "some query")

        assert result is None

    @patch("agents.nodes.github_issues.requests.get")
    @patch("agents.nodes.github_issues._repo_name_from_query", return_value="any-repo")
    def test_returns_none_when_user_fetch_fails(self, mock_name, mock_get):
        mock_get.return_value = _mock_response(401)

        result = _create_repo(FAKE_TOKEN, "some query")

        assert result is None

    @patch("agents.nodes.github_issues.requests.get")
    @patch("agents.nodes.github_issues.requests.post")
    @patch("agents.nodes.github_issues._repo_name_from_query", return_value="my-repo")
    def test_returns_none_on_500_error(self, mock_name, mock_post, mock_get):
        mock_get.return_value = self._make_user_resp()
        mock_post.return_value = _mock_response(500, text="Internal Server Error")

        result = _create_repo(FAKE_TOKEN, "some query")

        assert result is None

    @patch("agents.nodes.github_issues.requests.get")
    @patch("agents.nodes.github_issues.requests.post")
    @patch("agents.nodes.github_issues._repo_name_from_query", return_value="private-repo")
    def test_creates_private_repo(self, mock_name, mock_post, mock_get):
        mock_get.return_value = self._make_user_resp()
        mock_post.return_value = _mock_response(201, {"full_name": "testuser/private-repo"})

        _create_repo(FAKE_TOKEN, "some query")

        call_json = mock_post.call_args.kwargs["json"]
        assert call_json["private"] is True

    @patch("agents.nodes.github_issues.requests.get")
    @patch("agents.nodes.github_issues.requests.post")
    @patch("agents.nodes.github_issues._repo_name_from_query", return_value="my-repo")
    def test_description_includes_query(self, mock_name, mock_post, mock_get):
        mock_get.return_value = self._make_user_resp()
        mock_post.return_value = _mock_response(201, {"full_name": "testuser/my-repo"})

        _create_repo(FAKE_TOKEN, "Build an AI assistant")

        call_json = mock_post.call_args.kwargs["json"]
        assert "Build an AI assistant" in call_json["description"]


# ---------------------------------------------------------------------------
# github_issues_node
# ---------------------------------------------------------------------------

class TestGithubIssuesNode:
    BASE_STATE = {
        "query": "Build an AI assistant",
        "tasks": [],
        "create_repo": False,
        "github_issues": [],
        "messages": [],
        "local_result": {},
        "web_result": {},
        "summary": "",
        "mode": "project",
        "use_calendar": False,
        "readme": "",
        "notes": "",
        "calendar_events": [],
        "intent": "research",
    }

    def _state(self, **kwargs):
        return {**self.BASE_STATE, **kwargs}

    def test_returns_empty_when_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            os.environ.pop("GITHUB_REPO", None)
            result = github_issues_node(self._state())
        assert result == {"github_issues": []}

    @patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN})
    def test_returns_empty_when_no_repo_and_no_create_flag(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN}):
            os.environ.pop("GITHUB_REPO", None)
            result = github_issues_node(self._state(create_repo=False))
        assert result == {"github_issues": []}

    @patch("agents.nodes.github_issues.requests.post")
    def test_creates_issue_from_task(self, mock_post):
        task = {"title": "Set up CI", "description": "Configure GitHub Actions", "priority": "high", "duration_minutes": 60}
        issue_resp = _mock_response(201, {
            "number": 1,
            "title": "Set up CI",
            "html_url": "https://github.com/user/repo/issues/1",
        })
        mock_post.return_value = issue_resp

        with patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN, "GITHUB_REPO": "user/repo"}):
            result = github_issues_node(self._state(tasks=[task]))

        assert len(result["github_issues"]) == 1
        assert result["github_issues"][0]["number"] == 1
        assert result["github_issues"][0]["title"] == "Set up CI"

    @patch("agents.nodes.github_issues.requests.post")
    def test_issue_labels_include_priority_and_duration(self, mock_post):
        task = {"title": "Task", "description": "Desc", "priority": "medium", "duration_minutes": 30}
        mock_post.return_value = _mock_response(201, {
            "number": 2,
            "title": "Task",
            "html_url": "https://github.com/user/repo/issues/2",
        })

        with patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN, "GITHUB_REPO": "user/repo"}):
            github_issues_node(self._state(tasks=[task]))

        posted_json = mock_post.call_args.kwargs["json"]
        assert "time:30min" in posted_json["labels"]
        assert "priority:medium" in posted_json["labels"]

    @patch("agents.nodes.github_issues.requests.post")
    def test_issue_labels_empty_when_no_metadata(self, mock_post):
        task = {"title": "Task", "description": "Desc"}
        mock_post.return_value = _mock_response(201, {
            "number": 3,
            "title": "Task",
            "html_url": "https://github.com/user/repo/issues/3",
        })

        with patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN, "GITHUB_REPO": "user/repo"}):
            github_issues_node(self._state(tasks=[task]))

        posted_json = mock_post.call_args.kwargs["json"]
        assert posted_json["labels"] == []

    @patch("agents.nodes.github_issues.requests.post")
    def test_skips_failed_issues(self, mock_post):
        tasks = [
            {"title": "Task A", "description": "A"},
            {"title": "Task B", "description": "B"},
        ]
        mock_post.side_effect = [
            _mock_response(201, {"number": 1, "title": "Task A", "html_url": "https://github.com/user/repo/issues/1"}),
            _mock_response(403, text="Forbidden"),
        ]

        with patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN, "GITHUB_REPO": "user/repo"}):
            result = github_issues_node(self._state(tasks=tasks))

        assert len(result["github_issues"]) == 1
        assert result["github_issues"][0]["title"] == "Task A"

    @patch("agents.nodes.github_issues._create_repo", return_value="user/new-repo")
    @patch("agents.nodes.github_issues.requests.post")
    def test_creates_repo_when_flag_set(self, mock_post, mock_create):
        task = {"title": "Task", "description": "Desc"}
        mock_post.return_value = _mock_response(201, {
            "number": 1,
            "title": "Task",
            "html_url": "https://github.com/user/new-repo/issues/1",
        })

        with patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN}):
            os.environ.pop("GITHUB_REPO", None)
            result = github_issues_node(self._state(tasks=[task], create_repo=True, query="Build AI app"))

        mock_create.assert_called_once_with(FAKE_TOKEN, "Build AI app")
        assert result["github_issues"][0]["repo"] == "user/new-repo"

    @patch("agents.nodes.github_issues._create_repo", return_value=None)
    def test_returns_empty_when_repo_creation_fails(self, mock_create):
        with patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN}):
            os.environ.pop("GITHUB_REPO", None)
            result = github_issues_node(self._state(create_repo=True))

        assert result == {"github_issues": []}

    @patch("agents.nodes.github_issues.requests.post")
    def test_issue_contains_repo_reference(self, mock_post):
        task = {"title": "Write docs", "description": "Document the API"}
        mock_post.return_value = _mock_response(201, {
            "number": 5,
            "title": "Write docs",
            "html_url": "https://github.com/user/repo/issues/5",
        })

        with patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN, "GITHUB_REPO": "user/repo"}):
            result = github_issues_node(self._state(tasks=[task]))

        assert result["github_issues"][0]["repo"] == "user/repo"

    @patch("agents.nodes.github_issues.requests.post")
    def test_returns_empty_list_when_no_tasks(self, mock_post):
        with patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN, "GITHUB_REPO": "user/repo"}):
            result = github_issues_node(self._state(tasks=[]))

        mock_post.assert_not_called()
        assert result == {"github_issues": []}
