from agents import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from datetime import datetime

_TASK_SCHEMA = """Return ONLY a valid JSON list, no other text:
[
    {{
        "title": "...",
        "description": "...",
        "deadline": "YYYY-MM-DD",
        "start_time": "09:00",
        "priority": "high",
        "duration_minutes": 60
    }}
]

Each task must have:
- title: short task name
- description: what exactly to do
- deadline: specific date (YYYY-MM-DD format)
- start_time: suggested start time (HH:MM, 24h). Use realistic working hours, space tasks throughout the day.
- priority: high / medium / low
- duration_minutes: estimated time needed

Tasks must be written in the same language as the user query."""

RESEARCH_PLAN_TEMPLATE = """You are a task planner. Based on the research summary below, create an actionable plan for the user's goal.
If the summary is empty or contains insufficient information, return an empty list []. Do not invent tasks.
Today's date: {today}

Research summary:
{content}

User's goal: {query}

""" + _TASK_SCHEMA

NOTES_PLAN_TEMPLATE = """You are a study planner. Based on the learning notes below, create a concrete study plan for the user's goal.
Each task should correspond to a specific topic or section from the notes — do not invent topics not present in them.
If the notes are empty or contain no actionable material, return an empty list [].
Today's date: {today}

Learning notes:
{content}

User's goal: {query}

""" + _TASK_SCHEMA


def _run_planner(template: str, content: str, query: str) -> list[dict]:
    llm = get_llm(task="task_planner")
    chain = PromptTemplate(template=template, input_variables=["today", "content", "query"]) | llm | JsonOutputParser()
    tasks = chain.invoke({"today": datetime.now().strftime("%Y-%m-%d"), "content": content, "query": query})
    print([task for task in tasks])
    return tasks


def plan_task(summary: str, query: str) -> list[dict]:
    return _run_planner(RESEARCH_PLAN_TEMPLATE, summary, query)


def plan_task_from_notes(notes: str, query: str) -> list[dict]:
    return _run_planner(NOTES_PLAN_TEMPLATE, notes, query)