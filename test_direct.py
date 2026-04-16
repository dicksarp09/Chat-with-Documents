import sys

sys.path.insert(0, ".")

import requests

BASE = "http://localhost:8000/api/v1"

print("=== Direct Test: Upload & Query PDF ===")
with open("22705iied.pdf", "rb") as f:
    r = requests.post(
        f"{BASE}/upload", files={"file": ("test.pdf", f, "application/pdf")}
    )

print(f"Upload Status: {r.status_code}")
result = r.json()
print(f"Result: {result}")

doc_id = result.get("dataset_id")
if doc_id:
    print(f"\n=== Querying doc_id: {doc_id} ===")
    r = requests.post(
        f"{BASE}/query",
        params={"dataset_id": doc_id, "query": "What is this document about?"},
    )
    print(f"Query Status: {r.status_code}")
    query_result = r.json()
    answer = query_result.get("response", {}).get("answer", "N/A")
    print(f"Answer: {answer[:300]}...")
