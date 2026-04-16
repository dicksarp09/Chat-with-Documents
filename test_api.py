import requests
import json

BASE = "http://localhost:8000/api/v1"

print("=== 1. Upload CSV ===")
with open("sample_sales_data.csv", "rb") as f:
    r = requests.post(
        f"{BASE}/upload", files={"file": ("sample_sales_data.csv", f, "text/csv")}
    )
print(f"Status: {r.status_code}")
csv_result = r.json()
print(json.dumps(csv_result, indent=2)[:800])
print()

if csv_result.get("success"):
    dataset_id = csv_result["dataset_id"]

    print("=== 2. Query CSV - Total Sales by Product ===")
    r = requests.post(
        f"{BASE}/query",
        params={
            "dataset_id": dataset_id,
            "query": "What are the total sales by product?",
        },
    )
    print(f"Status: {r.status_code}")
    result = r.json()
    answer = result.get("response", {}).get("chat", {}).get("answer", "N/A")
    print(f"Answer: {answer[:300]}")
    print()

    print("=== 3. Query CSV - Highest Sales Region ===")
    r = requests.post(
        f"{BASE}/query",
        params={
            "dataset_id": dataset_id,
            "query": "Which region has the highest sales?",
        },
    )
    print(f"Status: {r.status_code}")
    result = r.json()
    answer = result.get("response", {}).get("chat", {}).get("answer", "N/A")
    print(f"Answer: {answer[:300]}")
    print()

print("=== 4. List All Datasets ===")
r = requests.get(f"{BASE}/datasets")
print(f"Status: {r.status_code}")
datasets = r.json()
for ds in datasets.get("datasets", []):
    print(f"  - {ds['dataset_id']} ({ds['type']}): {ds['filename']}")
