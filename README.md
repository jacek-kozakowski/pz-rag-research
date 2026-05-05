## Set Up

### 1. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 2. Wypełnienie pliku .env
```bash
cp .env.example .env
```
Uzupełnij klucze zgodnie z [API_MANUAL](API_MANUAL.md). Wymagany jest co najmniej jeden klucz LLM (OpenAI lub Groq) lub lokalna Ollama.

### 3. Uruchomienie MinIO (Docker)
```bash
docker compose up -d
```
MinIO przechowuje wgrane pliki PDF/DOCX. Domyślne dane logowania z `.env.example` działają bez zmian.

### 4. (Opcjonalnie) Ollama jako fallback LLM
```bash
ollama pull mistral
```
Używane automatycznie gdy brak kluczy OpenAI i Groq.

### 5. Uruchomienie aplikacji

**UI (Streamlit):**
```bash
streamlit run ui/app.py
```

**CLI:**
```bash
python main.py
```

---

## Tryby działania

### Tryb Project
Generuje pełną dokumentację projektu na podstawie opisu.

```
research_agent → summarization → task_planner → scaffolding → github_issues → readme
```

Wyjścia: podsumowanie, plan zadań, szkielet plików projektu, GitHub issues, README.md

### Tryb Learning
Tworzy notatki edukacyjne z zaindeksowanych dokumentów lub z internetu.

```
detect_intent → local_files: notes
             → research: research_agent → summarization → task_planner → (calendar →)? notes
```

- **local_files** — zapytanie dotyczy zaindeksowanych plików (PDF/DOCX z MinIO) → notatki z własnych materiałów
- **research** — temat spoza bazy → research webowy + notatki

Wyjścia: notatki w Markdown, (opcjonalnie) wydarzenia w Google Calendar

---

## Struktura projektu

```
RAGResearch/
├── .env
├── requirements.txt
├── main.py                        # punkt wejścia CLI
│
├── agents/
│   ├── __init__.py               # get_llm() — fabryka modeli (OpenAI/Groq/Ollama)
│   ├── state.py                  # AgentState — wspólny stan grafu
│   ├── graph.py                  # build_project_graph(), build_learning_graph()
│   ├── code_supervisor.py
│   └── nodes/
│       ├── detect_intent.py      # klasyfikacja zapytania: local_files vs research
│       ├── research.py           # agent badawczy z narzędziami
│       ├── summarization.py      # podsumowanie + planowanie zadań
│       ├── notes.py              # generowanie notatek edukacyjnych
│       ├── scaffolding.py        # generowanie szkieletu projektu
│       ├── github_issues.py      # tworzenie GitHub issues
│       ├── readme.py             # generowanie README
│       └── calendar.py          # eksport do Google Calendar
│
├── rag/
│   ├── loader.py                 # ładowanie PDF/DOCX z MinIO
│   ├── splitter.py               # podział dokumentów na chunki
│   ├── vector_storage.py         # ChromaDB (zapis, wyszukiwanie, find_relevant_sources)
│   └── minio_storage.py          # operacje na plikach w MinIO
│
├── research/
│   ├── local_researcher.py       # RAG Q&A na ChromaDB
│   ├── web_researcher.py         # wyszukiwanie webowe (Tavily / DuckDuckGo)
│   ├── research_tools.py         # narzędzia LangChain
│   ├── query_planner.py
│   ├── topic_decomposition.py
│   ├── summarizer.py
│   └── exporter.py
│
├── code/
│   ├── code_tools.py
│   └── loader.py
│
├── ui/
│   ├── app.py                    # interfejs Streamlit
│   └── components/
│       ├── sidebar.py            # sidebar (upload plików, ustawienia)
│       └── styles.py
│
├── tests/
│   └── test_github_issues.py
│
└── chroma_research/              # baza wektorowa ChromaDB (generowana automatycznie)
```

---

## Przepływ danych

```
użytkownik (prompt + pliki PDF/DOCX)
        ↓
    [Streamlit UI / CLI]
        ↓
   detect_intent
        ├── local_files ──→ ChromaDB (RAG z własnych plików) ──→ notes
        └── research ─────→ local_researcher (ChromaDB)
                           web_researcher (Tavily/DuckDuckGo)
                                ↓
                          summarization → task_planner
                                ↓                 ↓
                          (calendar)          scaffolding
                                ↓             github_issues
                            notes              readme
```

---

## Dostawcy LLM (priorytet)

| Priorytet | Dostawca | Klucz | Modele |
|-----------|----------|-------|--------|
| 1 | OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` (default), `gpt-4o` (notatki, planer) |
| 2 | Groq | `GROQ_API_KEY` | `llama-3.1-8b-instant` (default), `llama-3.3-70b-versatile` (notatki) |
| 3 | Ollama | — | `mistral` (lokalnie) |