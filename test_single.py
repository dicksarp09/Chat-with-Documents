import sys

sys.path.insert(0, ".")

from compression.compressor import CompressedContext
from llm.reasoning import get_reasoning_pipeline

reasoning = get_reasoning_pipeline()

context_text = """
Dickson Sarpong is an AI Engineer Intern at VoiceBreeze AI from November 2025 to February 2026.
He developed low-latency, two-way conversational agents using Gemini Live API, ElevenLabs, and Deepgram.
His skills include Voice & Audio (Gemini Live API, ElevenLabs, Deepgram, Whisper, TTS Fine-tuning),
LLMs (Function Calling, RAG, Prompt Engineering), AI Systems (Agentic Frameworks, LangChain, LlamaIndex),
ML infrastructure (Kubernetes, Docker, Terraform), and Deployment & Monitoring.
"""

context = CompressedContext(
    text=context_text,
    source_nodes=["test-node"],
    compression_ratio=1.0,
    original_length=len(context_text),
    compressed_length=len(context_text),
)

question = "Who is Dickson Sarpong and what is his experience?"

print("Question:", question)
print("\nContext length:", len(context_text))

result = reasoning.query_analysis(question, context)

print("\nAnswer:")
print(result.answer)
