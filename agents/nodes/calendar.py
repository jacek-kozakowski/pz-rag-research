import os
from datetime import datetime, timedelta

from agents.state import AgentState


def calendar_node(state: AgentState) -> AgentState:
    print("Calendar node executing...")

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        print("google-api-python-client not installed, skipping calendar events")
        return {"calendar_events": []}

    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")

    if not credentials_path or not os.path.exists(credentials_path):
        print("GOOGLE_CREDENTIALS_PATH not set or file not found, skipping calendar events")
        return {"calendar_events": []}

    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    service = build("calendar", "v3", credentials=creds)

    created_events = []
    start_date = datetime.utcnow()

    for task in state.get('tasks', []):
        duration_minutes = int(task.get('duration_minutes', 60))
        end_date = start_date + timedelta(minutes=duration_minutes)

        result = service.events().insert(
            calendarId=calendar_id,
            body={
                "summary": task.get('title', ''),
                "description": task.get('description', ''),
                "start": {"dateTime": start_date.isoformat() + "Z", "timeZone": "UTC"},
                "end": {"dateTime": end_date.isoformat() + "Z", "timeZone": "UTC"},
            }
        ).execute()

        created_events.append({
            "id": result["id"],
            "title": task.get('title', ''),
            "start": start_date.isoformat(),
            "url": result.get("htmlLink", "")
        })
        print(f"Created calendar event: {task.get('title', '')}")

        # next task starts 1 hour after previous ends
        start_date = end_date + timedelta(hours=1)

    return {"calendar_events": created_events}
