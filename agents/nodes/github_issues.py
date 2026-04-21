import base64
import json
import os
import re
import requests
from langchain_core.prompts import PromptTemplate

from agents import get_llm
from agents.state import AgentState

GITHUB_API = "https://api.github.com"

GENERATE_REPO_NAME_PROMPT = """Generate a short GitHub repository name (slug) for this project idea: {query}
Rules:
Only lowercase letters, digits, and hyphens
Max 40 characters
No prefix like 'ai-' or 'repo-'
English only
Return ONLY the slug, nothing else
"""

ISSUES_FROM_SCAFFOLD_PROMPT = """You are a project manager creating GitHub issues for a software project.

The project already has a scaffold — source files with stub implementations have been created.
Your job is to create issues for the work that STILL needs to be done: filling in the stubs, writing tests, wiring up integrations, deployment setup, configuration, etc.

Project summary:
{summary}

Scaffold files already created:
{scaffold_overview}

Generate GitHub issues in the same language as the project summary.
Each issue should describe concrete work a developer needs to do — NOT re-implement what the scaffold already provides.

Good issue examples: "Write unit tests for auth module", "Implement JWT token validation in auth.py", "Set up database migrations", "Configure environment variables for production".
Bad issue examples: "Create auth module", "Implement login" (already scaffolded).

Return a JSON list. Each element must have exactly:
  "title"       – short issue title
  "description" – 2-4 sentences describing the specific work to do
  "priority"    – high / medium / low

Return ONLY valid JSON list, no other text."""
def _get_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }


def _repo_name_from_query(query: str) -> str:
    """Uses LLM to generate a concise slug-style repo name from the query."""
    llm = get_llm()
    prompt = PromptTemplate(
        template=GENERATE_REPO_NAME_PROMPT,
        input_variables=["query"]
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    slug = response.content.strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')
    return slug[:40] or "ai-research-project"


def _repo_exists(token: str, full_name: str) -> bool:
    """Check if a repo actually exists on GitHub."""
    resp = requests.get(f"{GITHUB_API}/repos/{full_name}", headers=_get_headers(token))
    return resp.status_code == 200


def _create_repo(token: str, query: str) -> str | None:
    """Creates a new private GitHub repo. Returns 'owner/repo' or None on failure."""
    repo_name = _repo_name_from_query(query)

    user_resp = requests.get(f"{GITHUB_API}/user", headers=_get_headers(token))
    if user_resp.status_code != 200:
        print(f"[GitHub] Failed to fetch user: {user_resp.status_code}")
        return None
    owner = user_resp.json()["login"]
    full_name = f"{owner}/{repo_name}"

    resp = requests.post(
        f"{GITHUB_API}/user/repos",
        json={
            "name": repo_name,
            "description": re.sub(r'[\x00-\x1f\x7f]', ' ', f"Auto-generated project repo for: {query}")[:350].strip(),
            "private": True,
            "auto_init": True
        },
        headers=_get_headers(token)
    )

    if resp.status_code == 201:
        full_name = resp.json()["full_name"]
        print(f"[GitHub] Repo created: {full_name}")
        return full_name
    elif resp.status_code == 422:
        print(f"[GitHub] 422 on create — checking if repo '{full_name}' actually exists...")
        if _repo_exists(token, full_name):
            print(f"[GitHub] Repo confirmed exists: {full_name}")
            return full_name
        else:
            print(f"[GitHub] Repo does NOT exist despite 422. Response: {resp.text}")
            return None
    else:
        print(f"[GitHub] Failed to create repo: {resp.status_code} {resp.text}")
        return None


def _generate_issues_from_scaffold(summary: str, scaffold: list[dict]) -> list[dict]:
    scaffold_overview = "\n".join(
        f"- {e['filepath']}: {e.get('purpose', '')}" for e in scaffold
    )
    llm = get_llm()
    prompt = PromptTemplate(
        template=ISSUES_FROM_SCAFFOLD_PROMPT,
        input_variables=["summary", "scaffold_overview"]
    )
    response = (prompt | llm).invoke({"summary": summary, "scaffold_overview": scaffold_overview})
    raw = re.sub(r'^```(?:json)?\s*', '', response.content.strip())
    raw = re.sub(r'\s*```$', '', raw)
    try:
        issues = json.loads(raw)
        if not isinstance(issues, list):
            return []
        return issues
    except json.JSONDecodeError:
        print(f"[GitHub Issues] Failed to parse LLM JSON: {raw[:200]}")
        return []


def github_issues_node(state: AgentState) -> AgentState:
    print("GitHub issues node executing...")
    token = os.getenv("GITHUB_TOKEN")

    if not token:
        print("GITHUB_TOKEN not set, skipping GitHub issues creation")
        return {"github_issues": []}

    repo = os.getenv("GITHUB_REPO")

    if not repo and state.get('create_repo'):
        repo = _create_repo(token, state['query'])

    if not repo:
        print("No GITHUB_REPO set and repo creation skipped, skipping GitHub issues creation")
        return {"github_issues": []}

    scaffold = state.get('scaffold', [])
    summary = state.get('summary', '')

    if scaffold:
        issue_specs = _generate_issues_from_scaffold(summary, scaffold)
    else:
        issue_specs = [
            {"title": t.get('title', ''), "description": t.get('description', ''), "priority": t.get('priority', 'medium')}
            for t in state.get('tasks', [])
        ]

    headers = _get_headers(token)
    created_issues = []

    for spec in issue_specs:
        priority = spec.get('priority', 'medium')
        resp = requests.post(
            f"{GITHUB_API}/repos/{repo}/issues",
            json={
                "title": spec.get('title', ''),
                "body": spec.get('description', ''),
                "labels": [f"priority:{priority}"]
            },
            headers=headers
        )

        if resp.status_code == 201:
            data = resp.json()
            created_issues.append({
                "number": data["number"],
                "title": data["title"],
                "url": data["html_url"],
                "repo": repo,
            })
            print(f"[GitHub] Issue #{data['number']} created: {data['title']}")
        else:
            print(f"[GitHub] Failed to create issue '{spec.get('title', '')}': {resp.status_code} — {resp.text}")

    if created_issues and scaffold:
        repo = created_issues[0]['repo']
        pushed_files = []
        for entry in scaffold:
            filepath = entry.get('filepath', '')
            code = entry.get('code', '')
            if not filepath or not code:
                continue
            file_url = _push_file_to_repo(token, repo, filepath, code)
            pushed_files.append((filepath, file_url, entry.get('purpose', '')))
            print(f"[Scaffold] {'Pushed' if file_url else 'Failed'} {filepath}")

        if pushed_files:
            lines = ["## Project scaffold committed\n"]
            for filepath, url, purpose in pushed_files:
                lines.append(f"- [`{filepath}`]({url}) — {purpose}" if url else f"- `{filepath}` — {purpose}")
            resp = requests.post(
                f"{GITHUB_API}/repos/{repo}/issues/{created_issues[0]['number']}/comments",
                json={"body": "\n".join(lines)},
                headers=headers
            )
            if resp.status_code == 201:
                print(f"[GitHub] Scaffold summary posted on issue #{created_issues[0]['number']}")

    return {"github_issues": created_issues}


def _push_file_to_repo(token: str, repo: str, filepath: str, code: str) -> str | None:
    """Commits a file to the repo via Contents API. Returns the file's HTML URL or None."""
    content_b64 = base64.b64encode(code.encode()).decode()
    resp = requests.put(
        f"{GITHUB_API}/repos/{repo}/contents/{filepath}",
        json={
            "message": f"scaffold: add {filepath}",
            "content": content_b64,
        },
        headers=_get_headers(token)
    )
    if resp.status_code in (200, 201):
        return resp.json().get("content", {}).get("html_url")
    print(f"[Scaffold] Failed to push {filepath}: {resp.status_code} — {resp.text}")
    return None


