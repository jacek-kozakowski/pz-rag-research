## Struktura projektu
```
multi-agent-research/
│
├── .env                          # klucze API (OPENAI_API_KEY itp.)
├── requirements.txt
├── README.md
│
├── main.py                       # punkt wejścia - odpala cały pipeline
│
├── agents/
│   ├── __init__.py
│   ├── local_researcher.py       # przeszukuje ChromaDB (RAG)
│   ├── web_researcher.py         # przeszukuje internet (DuckDuckGo)
│   ├── summarizer.py             # łączy wyniki obu researcherów
│   ├── planner.py                # tworzy listę zadań z estymacją czasu
│   └── integrator.py            # eksportuje plan do .ics
│
├── rag/
│   ├── __init__.py
│   ├── loader.py                 # ładuje PDF/DOCX/TXT przez unstructured
│   ├── splitter.py               # dzieli tekst na chunki
│   └── vectorstore.py            # tworzy i odpytuje ChromaDB
│
├── calendar_export/
│   ├── __init__.py
│   └── ics_builder.py            # buduje plik .ics z listy zadań
│
├── ui/
│   └── app.py                    # interfejs Streamlit
│
├── data/
│   └── uploads/                  # tu użytkownik wrzuca swoje pliki
│
└── chroma_db/                    # baza wektorowa (generowana automatycznie)
```

---

## Przepływ MVP
```
użytkownik (prompt + pliki)
        ↓
    [Streamlit UI]
        ↓
  local_researcher  ←→  chroma_db (RAG z plików)
  web_researcher    ←→  DuckDuckGo
        ↓
    summarizer      →   spójny raport tekstowy
        ↓
     planner        →   lista zadań [{nazwa, czas, priorytet}]
        ↓
   integrator      