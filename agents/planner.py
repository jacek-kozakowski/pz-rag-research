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
- priority: high / medium / low
- duration_minutes: estimated time needed

The plan should be realistic, specific and directly related to the user's goal.
Return ONLY valid JSON list, no other text:
[
    {{
        "title": "...",
        "description": "...",
        "deadline": "YYYY-MM-DD",
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