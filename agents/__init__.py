from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import os

def get_llm(temperature: float = 0.0, task:str = "default"):
    """
    task = default => main llm for responses
    task = planner => llm for planning queries
    temperature = 0.0 => deterministic responses
    """
    if task == "planner":
        if _ollama_available():
            print("Using Ollama API")
            return ChatOllama(model="llama3.2", temperature=temperature)
        elif os.getenv("GROQ_API_KEY"):
            print("Using Groq API for planner")
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