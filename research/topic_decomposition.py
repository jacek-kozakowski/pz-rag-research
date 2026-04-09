from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents import get_llm
DECOMPOSE_TOPIC_PROMPT = """You are a precise assistant that extracts main topics from a user query.

Task:
Decompose the given question into 3 to 5 concise, specific topics.

Rules:
- Each topic must be directly related to the question.
- Avoid generic or vague topics (e.g., "technology", "science").
- Topics should be short noun phrases (not full sentences).
- Do not repeat similar topics.
- If the question is simple, still produce at least 3 meaningful subtopics.
- Return the topics in the same language as the user query.

Output format:
- Return ONLY a valid JSON array of strings.
- Do NOT include any explanations, text, or formatting outside JSON.

Now process this:
Question: {query}
"""


def decompose_topic(query: str) -> list[str]:
    llm = get_llm(task="query_planner")

    prompt = PromptTemplate(
        template=DECOMPOSE_TOPIC_PROMPT,
        input_variables=["query"]
    )
    chain = prompt | llm | JsonOutputParser()
    topics = chain.invoke({"query": query})
    print("Decomposed topics:", topics)
    return topics