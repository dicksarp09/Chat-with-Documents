import requests
import json

BASE = "http://localhost:8000/api/v1"

print("=== Upload PDF Document ===")
with open("22705iied.pdf", "rb") as f:
    r = requests.post(
        f"{BASE}/upload", files={"file": ("22705iied.pdf", f, "application/pdf")}
    )
print(f"Status: {r.status_code}")
doc_result = r.json()
print(json.dumps(doc_result, indent=2)[:1000])
print()

if doc_result.get("success"):
    doc_id = doc_result["dataset_id"]

    print("=== Query Document ===")
    r = requests.post(
        f"{BASE}/query",
        params={"dataset_id": doc_id, "query": "What is this document about?"},
    )
    print(f"Status: {r.status_code}")
    result = r.json()
    answer = result.get("response", {}).get("answer", "N/A")
    print(f"Answer: {answer[:500]}")
    print()

print("=== List All Datasets ===")
r = requests.get(f"{BASE}/datasets")
print(f"Status: {r.status_code}")
datasets = r.json()
for ds in datasets.get("datasets", []):
    print(f"  - {ds['dataset_id']} ({ds['type']}): {ds['filename']}")
