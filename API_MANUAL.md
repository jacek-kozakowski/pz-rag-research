# API Manual

This file describes all environment variables for the `.env` file (see `.env.example`).

---

## LLM Providers (at least one required)

The provider is selected automatically based on which key is set — **there is no `LLM_PROVIDER` variable**.
Priority: **OpenAI → Groq → Ollama**.

### [OpenAI](https://platform.openai.com)
```
OPENAI_API_KEY=
```
Best quality. Models used: `gpt-4o-mini` (default), `gpt-4o` (notes, task planner, code). Costs money.

### [Groq](https://console.groq.com/keys)
```
GROQ_API_KEY=
```
Free tier. Replaces OpenAI. Models used: `llama-3.1-8b-instant` (default), `llama-3.1-70b-versatile` (task planner), `llama-3.3-70b-versatile` (notes, code).

### Ollama (local, no key needed)
Fallback when neither OpenAI nor Groq is available. Requires `ollama pull mistral` and Ollama running locally.

---

## Embeddings

Selected automatically based on the available API key:

| API Key | `research` collection | `code` collection |
|---------|-----------------------|-------------------|
| `OPENAI_API_KEY` set | `text-embedding-3-small` (paid) | `text-embedding-3-small` (paid) |
| No OpenAI key | `intfloat/multilingual-e5-base` (HuggingFace, free) | `jinaai/jina-embeddings-v2-base-code` (HuggingFace, free) |

> ⚠ HuggingFace models are downloaded on first run.  
> ⚠ Mixing embedding models in the same ChromaDB collection causes a `ValueError` — delete the `chroma_research` / `chroma_code` directory and re-index if you switch providers.

---

## File Storage — MinIO (required)

MinIO runs via Docker (`docker compose up -d`). Default credentials from `.env.example` work out of the box.

```
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=password123
MINIO_BUCKET_NAME=rag-bucket
MINIO_ENDPOINT=http://localhost:9000
```

> ⚠ The variable is **`MINIO_BUCKET_NAME`** (not `MINIO_BUCKET`) — `rag/minio_storage.py` reads exactly that name. Default value when unset: `rag-bucket`.

---

## Web Search

### [Tavily](https://app.tavily.com/home)
```
TAVILY_API_KEY=
```
1000 free requests/month. Falls back to DuckDuckGo automatically when the key is not set or the quota is exceeded.

---

## GitHub Integration (optional)

```
GITHUB_TOKEN=
GITHUB_REPO=
```

- **`GITHUB_TOKEN`** — required for creating issues and pushing scaffold files. Generate at: GitHub → Settings → Developer settings → Personal access tokens. Required scope: `repo`.
- **`GITHUB_REPO`** — optional. Format: `owner/repo`. If set, the system uses this existing repo instead of creating a new one. If empty and "Create new repo" is checked in the UI, a repo is created automatically with an LLM-generated name.

---

## Google Calendar Integration (optional)

```
GOOGLE_CALENDAR_ID=your@email.com
GOOGLE_CREDENTIALS_PATH=/path/to/service_account.json
```

Requires a **service account** JSON (not an OAuth2 desktop flow):

1. Go to [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Enable **Google Calendar API**
2. Create a Service Account → download the key as a JSON file
3. **Share your calendar** with the service account email (e.g. `my-sa@project.iam.gserviceaccount.com`) granting "Make changes to events" permission
4. Set `GOOGLE_CREDENTIALS_PATH` to the path of the downloaded JSON
5. Set `GOOGLE_CALENDAR_ID` to your calendar email address (or `primary`)

> In the Streamlit UI you can upload the service account JSON directly via sidebar → ⚙️ API Settings — it will be saved to `~/.streamlit/google_credentials.json`.