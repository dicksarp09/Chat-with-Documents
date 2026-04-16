import requests

BASE = "http://localhost:8000/api/v1"

print("=== Upload PDF Document ===")
with open("22705iied.pdf", "rb") as f:
    r = requests.post(
        f"{BASE}/upload", files={"file": ("test.pdf", f, "application/pdf")}
    )
print(f"Status: {r.status_code}")
print(r.json())

print("\n=== Query Document ===")
doc_id = r.json().get("dataset_id")
if doc_id:
    r = requests.post(
        f"{BASE}/query",
        params={"dataset_id": doc_id, "query": "What is this document about?"},
    )
    print(f"Status: {r.status_code}")
    result = r.json()
    print(f"Answer: {result.get('response', {}).get('answer', 'N/A')[:500]}")
