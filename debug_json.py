import sys

sys.path.insert(0, ".")

from llm.groq_client import get_groq_client
import json
import re

groq = get_groq_client()

prompt = """Evaluate this:

Context: Dickson Sarpong is an AI Engineer with 5 years of experience.
Answer: Dickson Sarpong is an AI Engineer with 5 years of experience.

Return ONLY valid JSON with these exact fields:
- faithful (boolean)
- relevant (boolean)
- grounded (boolean)
- hallucinations (array)
- missing_info (array)
- issues (array)
- faithfulness_score (number 0-1)
- relevance_score (number 0-1)
- overall_score (number 0-1)
- explanation (string)

JSON:"""

response = groq.generate(prompt, system_prompt="Return ONLY JSON, no markdown or text.")
print("Raw response:")
print(repr(response))
print()

# Better parsing
cleaned = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
cleaned = re.sub(r"```\s*", "", cleaned)
cleaned = cleaned.strip()

print("Cleaned:")
print(repr(cleaned[:200]))
print()

# Find JSON object using a simpler approach
match = re.search(r"\{.+\}", cleaned, re.DOTALL)
if match:
    try:
        data = json.loads(match.group())
        print("Parsed:", data)
    except Exception as e:
        print("Parse error:", e)
else:
    print("No match found")
