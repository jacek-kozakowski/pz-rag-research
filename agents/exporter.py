import os
from datetime import datetime

def export_to_md(query: str, summary: str, plan: list) -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    filename = f"report_{date}.md"

    tasks_md = "\n".join([
        f"- [ ] **{t['title']}** (deadline: {t['deadline']}, {t['duration_minutes']} min)\n  {t['description']}"
        for t in plan
    ])

    content = f"""# Research Report
    **Query:** {query}  
    **Date:** {date}

    ---

    ## Summary

    {summary}

    ---

    ## Action Plan

    {tasks_md}
    """

    path = f"data/reports/{filename}"
    os.makedirs("data/reports", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

    print(f"Exported to {path}")
    return path