from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os

def get_llm(temperature: float = 0.0):
    # Temp 0 -> deterministic responses
    if os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI API")
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    elif os.getenv("GROQ_API_KEY"):
        print("Using Groq API")
        return ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
    else:
        raise ValueError("No API key found. Set OPENAI_API_KEY or GROQ_API_KEY environment variables.")
