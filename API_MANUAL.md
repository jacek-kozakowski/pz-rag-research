# API Manual
This file lists all environment variables for the `.env` file (see `.env.example`).

## LLM Providers (at least one required)

### [OpenAI](https://platform.openai.com)
```
OPENAI_API_KEY=
```
Best quality. Used for: `gpt-4o-mini` (default), `gpt-4o` (notes, task planner, code). Costs money.

### [Groq](https://console.groq.com/keys)
```
GROQ_API_KEY=
```
Free tier. Replaces OpenAI. Used for: `llama-3.1-8b-instant` (default), `llama-3.1-70b-versatile` (task planner), `llama-3.3-70b-versatile` (notes, code).

### Ollama (local, no key needed)
Fallback when neither OpenAI nor Groq key is set. Requires `ollama pull mistral` and Ollama running locally.

---

## Embeddings

Chosen automatically based on LLM provider:
- OpenAI key present → `text-embedding-3-small` (costs money)
- No OpenAI key → `intfloat/multilingual-e5-base` (HuggingFace, free, downloaded on first run)

---

## Storage — MinIO (required)

MinIO runs via Docker (`docker compose up -d`). Default credentials from `.env.example` work out of the box.

```
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=password123
MINIO_BUCKET=rag-documents
MINIO_ENDPOINT=http://localhost:9000
```

---

## Web Search

### [Tavily](https://app.tavily.com/home)
```
TAVILY_API_KEY=
```
1000 free requests/month. Falls back to DuckDuckGo automatically when not set or quota exceeded.

---

## GitHub Integration (optional)

```
GITHUB_TOKEN=
```
Required for **Project mode** GitHub issue creation and repo setup. Generate at GitHub → Settings → Developer settings → Personal access tokens. Needs `repo` scope.

---

## Google Calendar Integration (optional)

```
GOOGLE_CALENDAR_ID=your@email.com
GOOGLE_CREDENTIALS_PATH=/path/to/credentials.json
```
Required for **Learning mode** calendar export. 
1. Go to [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Enable **Google Calendar API**
2. Create OAuth 2.0 credentials → download as `credentials.json`
3. Set `GOOGLE_CREDENTIALS_PATH` to the path of that file