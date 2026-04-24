import sys
sys.path.insert(0, ".")

from app.engines.document_engine.wrapper import get_document_engine

engine = get_document_engine()

# Read PDF
with open('22705iied.pdf', 'rb') as f:
    pdf_bytes = f.read()

print('Processing PDF...')
result = engine.process(pdf_bytes, '22705iied.pdf')
print(f'Processed: {result}')

# Query the document
question = 'Who is at risk from food security issues?'
print(f'\nQuerying: {question}')
answer = engine.query(question, result['doc_id'])
print(f'\nAnswer: {answer.answer[:500] if len(answer.answer) > 500 else answer.answer}')
print(f'Key points: {len(answer.key_points)}')
print(f'Sources: {len(answer.sources)}')