import os
import pytest
from unittest.mock import patch, MagicMock

from agents.nodes.calendar import calendar_node


BASE_STATE = {
    "query": "plan my week",
    "messages": [],
    "local_result": {},
    "web_result": {},
    "summary": "",
    "tasks": [],
    "mode": "learning",
    "create_repo": False,
    "use_calendar": True,
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


SAMPLE_TASK = {
    "title": "Study algorithms",
    "description": "Review sorting algorithms",
    "deadline": "2026-06-01",
    "start_time": "10:00",
    "duration_minutes": 60,
}


class TestCalendarNodeNoCredentials:
    def test_returns_empty_when_credentials_path_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_CREDENTIALS_PATH", None)
            result = calendar_node(_state(tasks=[SAMPLE_TASK]))
        assert result == {"calendar_events": []}

    def test_returns_empty_when_credentials_file_missing(self):
        with patch.dict(os.environ, {"GOOGLE_CREDENTIALS_PATH": "/nonexistent/path.json"}):
            result = calendar_node(_state(tasks=[SAMPLE_TASK]))
        assert result == {"calendar_events": []}

    def test_returns_empty_list_not_none(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_CREDENTIALS_PATH", None)
            result = calendar_node(_state())
        assert isinstance(result["calendar_events"], list)


class TestCalendarNodeGoogleImportMissing:
    def test_returns_empty_when_google_packages_not_installed(self):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("google.oauth2", "googleapiclient.discovery"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = calendar_node(_state(tasks=[SAMPLE_TASK]))

        assert result == {"calendar_events": []}


def _make_service_mock(insert_data=None, insert_side_effect=None, calendar_access_error=None):
    """Build a mock Google Calendar service."""
    event_data = insert_data or {"id": "event123", "htmlLink": "https://calendar.google.com/event?eid=event123"}

    insert_mock = MagicMock()
    if insert_side_effect:
        insert_mock.execute.side_effect = insert_side_effect
    else:
        insert_mock.execute.return_value = event_data

    events_mock = MagicMock()
    events_mock.insert.return_value = insert_mock

    cal_info_mock = MagicMock()
    if calendar_access_error:
        cal_info_mock.execute.side_effect = calendar_access_error
    else:
        cal_info_mock.execute.return_value = {"summary": "Test Calendar"}

    calendars_mock = MagicMock()
    calendars_mock.get.return_value = cal_info_mock

    service = MagicMock()
    service.events.return_value = events_mock
    service.calendars.return_value = calendars_mock
    return service


def _fake_creds():
    return MagicMock(service_account_email="svc@test.iam.gserviceaccount.com")


class TestCalendarNodeWithMockedService:
    """
    google.oauth2 and googleapiclient are imported *inside* calendar_node, so we
    patch them at the source module level (not agents.nodes.calendar.*).
    """

    def _run(self, tasks, service, **env):
        env.setdefault("GOOGLE_CREDENTIALS_PATH", "/fake/creds.json")
        env.setdefault("GOOGLE_CALENDAR_ID", "test@group.calendar.google.com")

        sa_mock = MagicMock()
        sa_mock.Credentials.from_service_account_file.return_value = _fake_creds()

        with patch.dict(os.environ, env), \
             patch("os.path.exists", return_value=True), \
             patch("google.oauth2.service_account.Credentials.from_service_account_file", return_value=_fake_creds()), \
             patch("googleapiclient.discovery.build", return_value=service):
            return calendar_node(_state(tasks=tasks))

    def test_creates_event_for_each_task(self):
        service = _make_service_mock()
        result = self._run([SAMPLE_TASK], service)
        assert len(result["calendar_events"]) == 1
        assert result["calendar_events"][0]["id"] == "event123"
        assert result["calendar_events"][0]["title"] == "Study algorithms"

    def test_uses_fallback_duration_for_invalid_duration(self):
        service = _make_service_mock()
        task = {**SAMPLE_TASK, "duration_minutes": "not-a-number"}
        result = self._run([task], service)
        assert len(result["calendar_events"]) == 1

    def test_skips_task_on_insert_failure(self):
        service = _make_service_mock(insert_side_effect=Exception("API error"))
        result = self._run([SAMPLE_TASK, {**SAMPLE_TASK, "title": "Task 2"}], service)
        assert result["calendar_events"] == []

    def test_returns_empty_when_no_tasks(self):
        service = _make_service_mock()
        result = self._run([], service)
        assert result["calendar_events"] == []

    def test_uses_fallback_datetime_for_invalid_deadline(self):
        service = _make_service_mock()
        task = {**SAMPLE_TASK, "deadline": "not-a-date", "start_time": "bad-time"}
        result = self._run([task], service)
        assert len(result["calendar_events"]) == 1

    def test_returns_empty_when_calendar_access_fails(self):
        service = _make_service_mock(calendar_access_error=Exception("forbidden"))
        result = self._run([SAMPLE_TASK], service)
        assert result == {"calendar_events": []}