from agents import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from datetime import datetime

PROMPT_TEMPLATE = """You are a task planner. Based on the research summary, create an actionable plan.

Today's date: {today}

Research summary:
{summary}

User's goal: {query}

Create a plan as a JSON list of tasks. Each task must have:
- title: short task name
- description: what exactly to do
- deadline: specific date (YYYY-MM-DD format)
- start_time: suggested start time for this task (HH:MM, 24h format). Pick realistic working hours (e.g. 09:00, 10:30, 14:00). Space tasks sensibly throughout the day — avoid scheduling everything back-to-back.
- priority: high / medium / low
- duration_minutes: estimated time needed

The plan should be realistic, specific and directly related to the user's goal.
Task should be written in the same language as the user query.
Return ONLY valid JSON list, no other text:
[
    {{
        "title": "...",
        "description": "...",
        "deadline": "YYYY-MM-DD",
        "start_time": "09:00",
        "priority": "high",
        "duration_minutes": 60
    }}
]"""


def plan_task(summary: str, query: str) -> list[dict]:
    llm = get_llm(task="task_planner")

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["today", "summary", "query"]
    )

    chain = prompt | llm | JsonOutputParser()

    tasks = chain.invoke({"today" : datetime.now().strftime("%Y-%m-%d"), "summary": summary, "query": query })
    print([task for task in tasks])
    return tasks