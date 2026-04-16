import sys

sys.path.insert(0, ".")

import pandas as pd
from app.engines.csv_engine.processor import get_processor
from app.engines.csv_engine.profiler import profile_dataset


def test_csv_engine():
    print("=== CSV Intelligence Engine Test ===\n")

    csv_path = "sample_sales_data.csv"

    print(f"1. Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}\n")

    print("2. Testing profiler...")
    profile = profile_dataset(df, "test_001", "sample_sales_data.csv")
    print(f"   Profile shape: {profile.shape}")
    print(f"   Columns profiled: {len(profile.columns)}")
    for col in profile.columns[:3]:
        print(f"   - {col.name}: {col.type}")
    print()

    print("3. Testing CSV processor...")
    processor = get_processor()
    result = processor.upload_csv(csv_path, "sample_sales_data.csv")
    print(f"   Upload result: {result.status}")
    print(f"   Dataset ID: {result.dataset_id}")
    print(f"   Shape: {result.shape}")
    print()

    print("4. Testing query...")
    query_result = processor.process_query(
        result.dataset_id, "What are the total sales by product?"
    )
    print(f"   Query: What are the total sales by product?")
    print(f"   Answer: {query_result.chat.get('answer', 'N/A')[:200]}...")
    print(f"   Insights: {len(query_result.insights)}")
    print(f"   Tables: {len(query_result.tables)}")
    print(
        f"   Execution time: {query_result.metadata.get('execution_time_ms', 0):.0f}ms"
    )
    print()

    print("5. Testing another query...")
    query_result2 = processor.process_query(
        result.dataset_id, "Which region has the highest sales?"
    )
    print(f"   Query: Which region has the highest sales?")
    print(f"   Answer: {query_result2.chat.get('answer', 'N/A')[:200]}...")
    print()

    print("6. Testing dataset listing...")
    datasets = processor.list_datasets()
    print(f"   Total datasets: {len(datasets)}")
    for ds in datasets:
        print(f"   - {ds.dataset_id}: {ds.filename}")
    print()

    print("=== Test Complete ===")


if __name__ == "__main__":
    test_csv_engine()
