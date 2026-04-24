import sys
sys.path.insert(0, ".")

from app.engines.document_engine.wrapper import get_document_engine

engine = get_document_engine()

# Read PDF
with open('22705iied.pdf', 'rb') as f:
    pdf_bytes = f.read()

print('Processing PDF...')
result = engine.process(pdf_bytes, '22705iied.pdf')
print(f'Processed: doc_id={result["doc_id"]}, chunks={result["chunk_count"]}')

# Test query
question = 'Who is at risk from food security issues?'
print(f'\nQuery: {question}')
answer = engine.query(question, result['doc_id'])
print(f'\nAnswer: {answer.answer[:400]}...')
print(f'Key points: {len(answer.key_points)}')
print(f'Sources: {len(answer.sources)}')