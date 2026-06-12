# RAGResearch

Multi-agentowy system RAG zbudowany na LangGraph. Przyjmuje zapytanie użytkownika, bada je (web + lokalne dokumenty) i generuje ustrukturyzowany plan projektu lub notatki edukacyjne.

---

## Uruchomienie

### 1. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 2. Konfiguracja środowiska
```bash
cp .env.example .env
```
Uzupełnij klucze zgodnie z [API_MANUAL.md](API_MANUAL.md). Wymagany jest co najmniej jeden klucz LLM (OpenAI lub Groq) albo lokalna Ollama.

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

Tryb jest wykrywany automatycznie z zapytania przez `detect_mode` (klasyfikator LLM ze wstępnym sprawdzaniem słów kluczowych):
- **`project`** — użytkownik chce zbudować / stworzyć / wdrożyć projekt programistyczny
- **`learning`** — użytkownik chce zrozumieć, nauczyć się lub uzyskać notatki na dany temat

### Tryb Project

Pełny potok: research → podsumowanie → zadania → szkielet → GitHub issues → README.

```
research_agent → (pętla narzędzi) → summarization → task_planner → scaffolding → github_issues → readme → END
```

**Wyjścia:** podsumowanie badań, plan zadań, szkielet plików projektu (10–20 plików z kodem), GitHub issues + push scaffoldu, wygenerowany README.

### Tryb Learning

Generuje ustrukturyzowane notatki z zaindeksowanych dokumentów lub internetu.

```
detect_intent → local_files: notes → task_planner → END
             └→ research:   research_agent → summarization → task_planner → (calendar →)? notes → END
```

- **`local_files`** — zapytanie dotyczy zaindeksowanych plików kursowych → notatki generowane metodą map-reduce z pełnego tekstu dokumentów
- **`research`** — temat spoza lokalnej bazy → research webowy + notatki z zebranych danych

**Wyjścia:** notatki Markdown (z normalizacją LaTeX), opcjonalnie wydarzenia w Google Calendar.

---

## Struktura projektu

```
RAGResearch/
├── .env                            # sekrety (na podstawie .env.example)
├── requirements.txt
├── main.py                         # punkt wejścia CLI
│
├── agents/
│   ├── __init__.py                 # get_llm() — fabryka modeli (OpenAI / Groq / Ollama)
│   ├── state.py                    # AgentState — wspólny TypedDict LangGraph
│   ├── graph.py                    # build_project_graph(), build_learning_graph()
│   ├── code_supervisor.py          # supervisor LangGraph: code_writer + test_writer + doc_writer
│   └── nodes/
│       ├── detect_mode.py          # klasyfikacja: project vs learning (wywoływana przez UI przed grafem)
│       ├── detect_intent.py        # klasyfikacja zapytania learning: local_files vs research
│       ├── research.py             # ReAct agent badawczy (dekompozycja → lokalny → web)
│       ├── summarization.py        # summarization_node + task_planner_node
│       ├── notes.py                # notatki edukacyjne (map-reduce dla local_files, agentowy dla research)
│       ├── scaffolding.py          # wykrywanie stacku + generowanie szkieletu plików projektu
│       ├── github_issues.py        # tworzenie repo GitHub, issues, push scaffoldu
│       ├── readme.py               # generowanie README projektu
│       └── calendar.py             # eksport do Google Calendar (service account)
│
├── rag/
│   ├── loader.py                   # ładowanie PDF / DOCX / TXT (PyMuPDF, Docx2txt, TextLoader)
│   ├── splitter.py                 # podział dokumentów na nakładające się chunki
│   ├── vector_storage.py           # ChromaDB: dwie kolekcje (research, code); find_relevant_sources
│   └── minio_storage.py            # MinIO (S3): upload, download, lista, usuwanie, load_full_documents
│
├── research/
│   ├── local_researcher.py         # RAG Q&A na ChromaDB
│   ├── web_researcher.py           # wyszukiwanie webowe: Tavily (podstawowy) / DuckDuckGo (fallback)
│   ├── research_tools.py           # wrappery @tool LangChain: search_local, search_web, decompose_topic
│   ├── query_planner.py            # planowanie zapytań RAG i webowych z zapytania użytkownika
│   ├── topic_decomposition.py      # dekompozycja złożonego zapytania na podtematy
│   ├── summarizer.py               # scalanie wyników lokalnych i webowych w jedno podsumowanie
│   ├── planner.py                  # plan_task() / plan_task_from_notes() → lista zadań z terminami
│   └── exporter.py                 # (utility) helpery eksportu
│
├── code/
│   ├── code_tools.py               # generate_code / generate_tests / generate_documentation
│   └── loader.py                   # index_codebase: lokalna ścieżka / URL GitHub / ZIP → kolekcja code
│
├── ui/
│   ├── app.py                      # UI Streamlit ze streamingiem węzłów grafu na żywo
│   └── components/
│       ├── sidebar.py              # upload i indeksowanie plików, zarządzanie plikami, ustawienia API
│       └── styles.py               # własne style CSS
│
├── tests/                          # testy jednostkowe pytest (bez live API calls, wszystko mockowane)
│   ├── test_calendar_node.py
│   ├── test_detect_intent.py
│   ├── test_detect_mode.py
│   ├── test_e2e_project_flow.py
│   ├── test_github_issues.py
│   ├── test_notes_node.py
│   ├── test_readme_node.py
│   ├── test_research_nodes.py
│   ├── test_scaffolding_node.py
│   └── test_summarization_node.py
│
└── chroma_research/                # baza wektorowa ChromaDB (generowana automatycznie przy pierwszym indeksowaniu)
```

---

## Stan (`agents/state.py`)

`AgentState` to `TypedDict` LangGraph współdzielony przez wszystkie węzły:

| Pole | Typ | Opis |
|------|-----|------|
| `query` | `str` | Oryginalne zapytanie użytkownika |
| `messages` | `list` | Historia wiadomości LangChain (append-only przez `add_messages`) |
| `local_result` | `dict` | Wynik lokalnego wyszukiwania RAG (`answer`, `sources`) |
| `web_result` | `dict` | Wynik wyszukiwania webowego (`answer`, `source`) |
| `summary` | `str` | Wyjście `summarization_node` |
| `tasks` | `list[dict]` | Zadania z polami `title`, `description`, `deadline`, `start_time`, `priority`, `duration_minutes` |
| `mode` | `'project'\|'learning'` | Wykryty tryb potoku |
| `intent` | `'local_files'\|'research'` | Wykryty intent w trybie learning |
| `create_repo` | `bool` | Czy automatycznie tworzyć repo GitHub |
| `use_calendar` | `bool` | Czy eksportować zadania do Google Calendar |
| `github_issues` | `list[dict]` | Utworzone issues: `number`, `title`, `url`, `repo` |
| `scaffold` | `list[dict]` | Wygenerowane pliki: `filepath`, `purpose`, `code` |
| `language` | `str` | Główny język wykryty ze scaffoldu (np. `"Python"`) |
| `readme` | `str` | Wygenerowana treść README |
| `notes` | `str` | Wygenerowane notatki edukacyjne (Markdown) |
| `calendar_events` | `list[dict]` | Utworzone wydarzenia: `id`, `title`, `start`, `url` |
| `web_enabled` | `bool` | Czy wyszukiwanie webowe jest dozwolone (przełącznik w sidebarze UI) |

---

## Fabryka LLM (`agents/__init__.py`)

`get_llm(task, temperature)` zwraca najlepszy dostępny model dla danego zadania:

| Task | OpenAI | Groq | Ollama |
|------|--------|------|--------|
| `default` | `gpt-4o-mini` | `llama-3.1-8b-instant` | `mistral` |
| `query_planner` | `gpt-4o-mini` | `llama-3.1-8b-instant` | `mistral` |
| `task_planner` | `gpt-4o` | `llama-3.1-70b-versatile` | `mistral` |
| `notes` | `gpt-4o` | `llama-3.3-70b-versatile` | — |
| `code` | `gpt-4o` | `llama-3.3-70b-versatile` | — |

Priorytet dostawcy: **OpenAI → Groq → Ollama**. Rzuca `ValueError` gdy żaden nie jest dostępny.

---

## Potok RAG (`rag/`)

### Embeddingi (wybierane automatycznie)

| Dostawca | Kolekcja research | Kolekcja code |
|----------|-------------------|---------------|
| OpenAI (klucz ustawiony) | `text-embedding-3-small` | `text-embedding-3-small` |
| Brak klucza OpenAI | `intfloat/multilingual-e5-base` (HuggingFace) | `jinaai/jina-embeddings-v2-base-code` (HuggingFace) |

### Kolekcje ChromaDB
- **`research`** — dokumenty użytkownika (PDF, DOCX, TXT), przechowywane w `./chroma_research`
- **`code`** — pliki kodu źródłowego, przechowywane w `./chroma_code`

> ⚠ Mieszanie modeli embeddingów w tej samej kolekcji powoduje `ValueError`. Usuń katalog ChromaDB i przeindeksuj jeśli zmieniasz dostawcę.

### `find_relevant_sources` — dwurundowe wyszukiwanie
1. Runda 1: szerokie wyszukiwanie k=30, wybiera pliki z ≥10 trafieniami poniżej odległości cosinusowej 0.6 (dominujące)
2. Runda 2: ponowne wyszukiwanie z wykluczeniem dominujących plików, wyłania mniejsze pliki z ≥3 trafieniami
3. Sędzia LLM filtruje kandydatów czytając rzeczywiste fragmenty przed generowaniem notatek

---

## Wyszukiwanie webowe (`research/web_researcher.py`)

- **Tavily** — podstawowy (do 1000 darmowych żądań/miesiąc); wymaga `TAVILY_API_KEY`
- **DuckDuckGo** — automatyczny fallback gdy Tavily nie jest skonfigurowany lub wyczerpany limit

---

## Integracja z GitHub (`agents/nodes/github_issues.py`)

Wymaga `GITHUB_TOKEN` z uprawnieniem `repo`.

1. Opcjonalnie tworzy prywatne repo (nazwa generowana przez LLM z zapytania)
2. Generuje issues — na podstawie przeglądu scaffoldu (jeśli istnieje) lub listy zadań
3. Pushuje wszystkie pliki scaffoldu do repo przez Contents API
4. Zamieszcza komentarz z listą plików na pierwszym issue

Ustaw `GITHUB_REPO=owner/repo` w `.env` żeby użyć istniejącego repo zamiast tworzyć nowe.

---

## Integracja z Google Calendar (`agents/nodes/calendar.py`)

Wymaga **service account** JSON (nie OAuth2 desktop):

1. Włącz Google Calendar API w [Google Cloud Console](https://console.cloud.google.com)
2. Utwórz service account → pobierz JSON
3. Udostępnij kalendarz adresowi email service account
4. Ustaw `GOOGLE_CREDENTIALS_PATH` i `GOOGLE_CALENDAR_ID` w `.env`

---

## Supervisor kodu (`agents/code_supervisor.py`)

Niezależny supervisor LangGraph (nie część głównego grafu) z trzema sub-agentami:
- **`code_writer`** — używa `retrieve_code_context` (RAG na kolekcji code) + `write_code`
- **`test_writer`** — używa `write_tests`
- **`doc_writer`** — używa `write_documentation`

Kolejność: kod → testy → dokumentacja.

---

## UI Streamlit (`ui/`)

Live streaming węzłów grafu przez `graph.stream(..., stream_mode="updates")`.

**Funkcje sidebara:**
- Upload i indeksowanie plików (PDF, DOCX, TXT lub kod źródłowy) do kolekcji Research lub Code
- Indeksowanie całego folderu lokalnego po ścieżce
- Indeksowanie codebase (ścieżka lokalna, URL GitHub lub ZIP)
- Zarządzanie zaindeksowanymi plikami (lista + usuwanie per kolekcja)
- Przełącznik wyszukiwania webowego
- Konfiguracja kluczy API (zapisywane do `~/.streamlit/secrets.toml`)

**Wyjścia głównego obszaru:**

| Tryb | Wyświetlane wyjścia |
|------|---------------------|
| Project | Podsumowanie, plan zadań (jeśli brak issues), GitHub issues, drzewo plików scaffoldu + kod, pobieranie ZIP, README |
| Learning | Podsumowanie, rozwijane sekcje local/web research, plan zadań, wydarzenia kalendarza, notatki edukacyjne |

---

## Testy

```bash
pytest tests/
```

Wszystkie testy używają `pytest` z `unittest.mock` — bez live API calls. 10 plików testowych pokrywa wszystkie główne węzły oraz test end-to-end przepływu project.

---

## Zmienne środowiskowe

Pełna dokumentacja w [API_MANUAL.md](API_MANUAL.md). Skrótowy przegląd:

| Zmienna | Wymagana | Opis |
|---------|----------|------|
| `OPENAI_API_KEY` | jeśli OpenAI | Klucz OpenAI |
| `GROQ_API_KEY` | jeśli Groq | Klucz Groq |
| `TAVILY_API_KEY` | nie | Tavily web search (fallback: DuckDuckGo) |
| `MINIO_ENDPOINT` | tak | URL MinIO (domyślnie: `http://localhost:9000`) |
| `MINIO_ACCESS_KEY` | tak | Klucz dostępu MinIO |
| `MINIO_SECRET_KEY` | tak | Sekret MinIO |
| `MINIO_BUCKET_NAME` | tak | Nazwa bucketu (domyślnie: `rag-bucket`) |
| `GITHUB_TOKEN` | dla GitHub | PAT z uprawnieniem `repo` |
| `GITHUB_REPO` | nie | `owner/repo` — użyj istniejącego repo zamiast tworzyć nowe |
| `GOOGLE_CALENDAR_ID` | dla kalendarza | ID kalendarza (np. `twój@email.com`) |
| `GOOGLE_CREDENTIALS_PATH` | dla kalendarza | Ścieżka do JSON service account |