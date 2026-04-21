import base64
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

CODE_SNIPPET_PROMPT = """
Generate a concise starter code snippet in {language} for this task:
Title: {title}
Description: {description}
Provide only the code with minimal comments.
"""
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

    headers = _get_headers(token)
    created_issues = []

    scaffold_index = {
        entry['task_title']: entry
        for entry in state.get('scaffold', [])
        if 'task_title' in entry
    }

    for task in state.get('tasks', []):
        labels = []
        if duration := task.get('duration_minutes'):
            labels.append(f"time:{duration}min")
        if priority := task.get('priority'):
            labels.append(f"priority:{priority}")

        body = task.get('description', '')
        scaffold_entry = scaffold_index.get(task.get('title', ''))
        if scaffold_entry:
            body += f"\n\n---\n**Scaffold file:** `{scaffold_entry['filepath']}`"

        resp = requests.post(
            f"{GITHUB_API}/repos/{repo}/issues",
            json={
                "title": task.get('title', ''),
                "body": body,
                "labels": labels
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
                "task": task
            })
            print(f"[GitHub] Issue #{data['number']} created: {data['title']}")
        else:
            print(f"[GitHub] Failed to create issue '{task.get('title', '')}': {resp.status_code} — {resp.text}")

    return {"github_issues": created_issues}


def _push_file_to_repo(token: str, repo: str, filepath: str, code: str, issue_number: int) -> str | None:
    """Commits a file to the repo via Contents API. Returns the file's HTML URL or None."""
    content_b64 = base64.b64encode(code.encode()).decode()
    resp = requests.put(
        f"{GITHUB_API}/repos/{repo}/contents/{filepath}",
        json={
            "message": f"scaffold: add {filepath} (issue #{issue_number})",
            "content": content_b64,
        },
        headers=_get_headers(token)
    )
    if resp.status_code in (200, 201):
        return resp.json().get("content", {}).get("html_url")
    print(f"[Scaffold] Failed to push {filepath}: {resp.status_code} — {resp.text}")
    return None


def code_snippets_node(state: AgentState) -> AgentState:
    print("Code snippets node executing...")
    token = os.getenv("GITHUB_TOKEN")
    issues = state.get('github_issues', [])

    if not token or not issues:
        print("No token or no issues, skipping code snippets")
        return {}

    scaffold_index = {
        entry['task_title']: entry
        for entry in state.get('scaffold', [])
        if 'task_title' in entry
    }
    language = state.get('language') or "Python"

    llm = get_llm()

    for issue in issues:
        task = issue.get('task', {})
        repo = issue.get('repo') or os.getenv("GITHUB_REPO")
        if not repo:
            continue

        title = task.get('title', '')
        scaffold_entry = scaffold_index.get(title)

        if scaffold_entry:
            code = scaffold_entry['code']
            filepath = scaffold_entry['filepath']
            file_url = _push_file_to_repo(token, repo, filepath, code, issue['number'])
            if file_url:
                comment_body = f"Scaffold file committed: [{filepath}]({file_url})"
            else:
                lang_lower = language.lower()
                comment_body = f"## Scaffold: `{filepath}`\n\n```{lang_lower}\n{code}\n```"
        else:
            prompt = PromptTemplate(
                template=CODE_SNIPPET_PROMPT,
                input_variables=["language", "title", "description"]
            )
            chain = prompt | llm
            response = chain.invoke({"language": language, "title": title, "description": task.get('description', '')})
            lang_lower = language.lower()
            comment_body = f"## Starter Code Snippet\n\n```{lang_lower}\n{response.content}\n```"

        resp = requests.post(
            f"{GITHUB_API}/repos/{repo}/issues/{issue['number']}/comments",
            json={"body": comment_body},
            headers=_get_headers(token)
        )
        if resp.status_code == 201:
            print(f"[GitHub] Scaffold file pushed and linked on issue #{issue['number']}")
        else:
            print(f"Failed to add comment to issue #{issue['number']}: {resp.status_code}")

    return {}
