import requests
import json
import time

API_BASE = "http://localhost:8000/api/v1"

# Check stats
print("Checking system stats...")
resp = requests.get(f"{API_BASE}/stats")
stats = resp.json()
print(f"Stats: {json.dumps(stats, indent=2)}")

# Check documents
print("\nChecking documents...")
resp = requests.get(f"{API_BASE}/documents")
docs = resp.json()
print(f"Documents: {json.dumps(docs, indent=2)}")

if docs["total"] > 0:
    doc_id = docs["documents"][0]["doc_id"]

    # Check specific document
    print(f"\nChecking document {doc_id}...")
    resp = requests.get(f"{API_BASE}/document/{doc_id}")
    doc_info = resp.json()
    print(f"Doc info: {json.dumps(doc_info, indent=2)}")

    # Query
    print("\nRunning query...")
    query_data = {"query": "Dickson", "doc_id": doc_id}
    resp = requests.post(f"{API_BASE}/query", json=query_data)
    result = resp.json()
    print(f"Query result: {json.dumps(result, indent=2)}")
