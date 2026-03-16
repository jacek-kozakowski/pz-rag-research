from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import os

def get_llm(temperature: float = 0.0, task:str = "default"):
    """
    task = default => main llm for responses
    task = query_planner => llm for planning queries
    task = task_planner => llm for task planning
    temperature = 0.0 => deterministic responses
    """
    if task == "task_planner":
        temperature = min(max(temperature, 0.3), 0.99)
        if os.getenv("OPENAI_API_KEY"):
            print("Using OpenAI API for task planner")
            return ChatOpenAI(model="gpt-4o", temperature=temperature)
        elif os.getenv("GROQ_API_KEY"):
            print("Using Groq API for task planner")
            return ChatGroq(model="llama-3.1-70b-versatile", temperature=temperature)
        elif _ollama_available():
            print("Using Ollama API for task planner")
            return ChatOllama(model="mistral", temperature=temperature)

    if task == "query_planner":
        if _ollama_available():
            print("Using Ollama API for query planner")
            return ChatOllama(model="llama3.2", temperature=temperature)
        elif os.getenv("GROQ_API_KEY"):
            print("Using Groq API for query planner")
            return ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)

    if os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI API")
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    elif os.getenv("GROQ_API_KEY"):
        print("Using Groq API")
        return ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
    else:
        raise ValueError("No API key found. Set OPENAI_API_KEY or GROQ_API_KEY environment variables.")

def _ollama_available() -> bool:
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False