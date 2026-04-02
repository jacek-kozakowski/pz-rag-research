import os
from datetime import datetime, timedelta

from agents.state import AgentState


def calendar_node(state: AgentState) -> AgentState:
    print("Calendar node executing...")

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        print("[CALENDAR DEBUG] google-api-python-client not installed, skipping calendar events")
        return {"calendar_events": []}

    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")

    print(f"[CALENDAR DEBUG] GOOGLE_CALENDAR_ID={calendar_id!r}")
    print(f"[CALENDAR DEBUG] GOOGLE_CREDENTIALS_PATH={credentials_path!r}")

    if not credentials_path:
        print("[CALENDAR DEBUG] GOOGLE_CREDENTIALS_PATH not set, skipping")
        return {"calendar_events": []}

    if not os.path.exists(credentials_path):
        print(f"[CALENDAR DEBUG] credentials file not found at: {credentials_path}")
        return {"calendar_events": []}

    tasks = state.get('tasks', [])
    print(f"[CALENDAR DEBUG] tasks count: {len(tasks)}")
    for i, t in enumerate(tasks):
        print(f"[CALENDAR DEBUG] task[{i}]: {t}")

    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/calendar"]
        )
        print(f"[CALENDAR DEBUG] credentials loaded, service_account_email={creds.service_account_email!r}")
    except Exception as e:
        print(f"[CALENDAR DEBUG] failed to load credentials: {e}")
        return {"calendar_events": []}

    try:
        service = build("calendar", "v3", credentials=creds)
        print(f"[CALENDAR DEBUG] Google Calendar service built OK")
    except Exception as e:
        print(f"[CALENDAR DEBUG] failed to build service: {e}")
        return {"calendar_events": []}

    # verify calendar access
    try:
        cal_info = service.calendars().get(calendarId=calendar_id).execute()
        print(f"[CALENDAR DEBUG] calendar access OK: {cal_info.get('summary', calendar_id)!r}")
    except Exception as e:
        print(f"[CALENDAR DEBUG] cannot access calendar '{calendar_id}': {e}")
        print("[CALENDAR DEBUG] hint: share the calendar with the service account email above")
        return {"calendar_events": []}

    created_events = []
    start_date = datetime.utcnow()

    for task in tasks:
        try:
            duration_minutes = int(task.get('duration_minutes', 60))
        except (ValueError, TypeError) as e:
            print(f"[CALENDAR DEBUG] invalid duration_minutes={task.get('duration_minutes')!r}, using 60: {e}")
            duration_minutes = 60

        end_date = start_date + timedelta(minutes=duration_minutes)

        event_body = {
            "summary": task.get('title', ''),
            "description": task.get('description', ''),
            "start": {"dateTime": start_date.isoformat() + "Z", "timeZone": "UTC"},
            "end": {"dateTime": end_date.isoformat() + "Z", "timeZone": "UTC"},
        }
        print(f"[CALENDAR DEBUG] inserting event: {event_body['summary']!r} start={event_body['start']['dateTime']}")

        try:
            result = service.events().insert(
                calendarId=calendar_id,
                body=event_body
            ).execute()
            print(f"[CALENDAR DEBUG] event created, id={result.get('id')!r}, link={result.get('htmlLink')!r}")
        except Exception as e:
            print(f"[CALENDAR DEBUG] failed to insert event '{task.get('title', '')}': {e}")
            continue

        created_events.append({
            "id": result["id"],
            "title": task.get('title', ''),
            "start": start_date.isoformat(),
            "url": result.get("htmlLink", "")
        })

        # next task starts 1 hour after previous ends
        start_date = end_date + timedelta(hours=1)

    print(f"[CALENDAR DEBUG] done, created {len(created_events)} events")
    return {"calendar_events": created_events}
