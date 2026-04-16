import requests

BASE = "http://localhost:8000/api/v1"

print("=== Debug: Check Vector Store ===")
r = requests.get(f"{BASE}/datasets")
print(f"Status: {r.status_code}")
datasets = r.json()
for ds in datasets.get("datasets", []):
    print(f"  - {ds['dataset_id']} ({ds['type']}): {ds['filename']}")

print("\n=== Test PDF Query ===")
doc_id = "doc_660391377899"
r = requests.post(
    f"{BASE}/query",
    params={"dataset_id": doc_id, "query": "What is the main topic of this document?"},
)
print(f"Status: {r.status_code}")
result = r.json()
print(f"Response keys: {result.get('response', {}).keys()}")
print(f"Answer: {result.get('response', {}).get('answer', 'N/A')[:500]}")
