import sys

sys.path.insert(0, ".")

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


def test_unified_upload():
    print("=== Testing Unified Upload ===\n")

    print("1. Uploading CSV file...")
    with open("sample_sales_data.csv", "rb") as f:
        files = {"file": ("sample_sales_data.csv", f, "text/csv")}
        response = requests.post(f"{BASE_URL}/upload", files=files)

    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Response: {json.dumps(result, indent=2)[:500]}...")

    if result.get("success"):
        dataset_id = result["dataset_id"]
        print(f"\n2. Querying dataset: {dataset_id}")

        query_response = requests.post(
            f"{BASE_URL}/query",
            params={
                "dataset_id": dataset_id,
                "query": "What are the total sales by product?",
            },
        )
        print(f"   Status: {query_response.status_code}")
        query_result = query_response.json()
        print(
            f"   Answer: {query_result.get('response', {}).get('chat', {}).get('answer', 'N/A')[:200]}..."
        )

    print("\n3. Listing all datasets...")
    list_response = requests.get(f"{BASE_URL}/datasets")
    print(f"   Status: {list_response.status_code}")
    datasets = list_response.json()
    print(f"   Total datasets: {datasets.get('total', 0)}")
    for ds in datasets.get("datasets", []):
        print(f"   - {ds['dataset_id']} ({ds['type']}): {ds['filename']}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_unified_upload()
