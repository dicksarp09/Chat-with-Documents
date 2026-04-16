import sys

sys.path.insert(0, ".")

from llm.groq_client import get_groq_client

groq = get_groq_client()

LLM_JUDGE_PROMPT = """You are an expert evaluator for RAG systems.

Given:
- The original question
- The context that was used to generate the answer
- The AI-generated answer

Evaluate the answer on the following criteria:
1. FAITHFULNESS: Is the answer grounded in the provided context?
2. RELEVANCE: Does the answer directly address the question?
3. GROUNDEDNESS: Are the claims verifiable from the context?

Provide your evaluation in JSON format ONLY with these exact fields:
{
    "faithful": true/false,
    "relevant": true/false,
    "grounded": true/false,
    "hallucinations": [],
    "missing_info": [],
    "issues": [],
    "faithfulness_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "explanation": "brief explanation"
}

Question: Who is Dickson Sarpong?

Context:
Dickson Sarpong is an AI Engineer with experience in building automated repository analysis systems and implementing governance and safety mechanisms for LLM interaction.

Answer:
Dickson Sarpong is an AI Engineer who works on repository analysis and LLM safety.

JSON:"""

response = groq.generate(
    LLM_JUDGE_PROMPT, system_prompt="Return ONLY JSON, no markdown or text."
)
print("Raw response:")
print(repr(response))
print()
print("Response:")
print(response)
