import sys
sys.path.insert(0, ".")

from app.engines.document_engine.wrapper import get_document_engine
from storage.vector_store import get_vector_store
from storage.sqlite_store import get_sqlite_store
from embeddings.embedder import get_embedder

engine = get_document_engine()

# Load existing doc from SQLite
sqlite = get_sqlite_store()
datasets = sqlite.get_all_datasets()
print(f"Found {len(datasets)} datasets in SQLite")
for ds in datasets:
    print(f"  - {ds['name']}: {ds['id']}")

# Get the doc_id from existing dataset
if datasets:
    doc_id = datasets[0]['id']
    print(f"\nUsing existing document: {datasets[0]['name']} (ID: {doc_id})")

    # Test query
    questions = [
        "Who is at risk from food security issues?",
        "What are the main causes of food insecurity?",
        "What recommendations does the document make?"
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print('-'*60)
        answer = engine.query(q, doc_id)
        print(f"A: {answer.answer[:300]}...")
        print(f"Key points: {len(answer.key_points)}")
        print(f"Sources: {len(answer.sources)}")