import sys

sys.path.insert(0, ".")

from parsers.pdf_parser import parse_pdf
from chunking.hierarchical_chunker import chunk_document
from embeddings.embedder import get_embedder
from storage.vector_store import get_vector_store
from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.reranker import get_reranker
from compression.compressor import get_compressor
from llm.reasoning import get_reasoning_pipeline
from evaluation.evaluator import EvaluationPipeline, TestCase


def run_query_pipeline(question: str):
    """Run the full query pipeline and return context + answer"""
    retriever = get_hybrid_retriever()
    reranker = get_reranker()
    compressor = get_compressor()
    reasoning = get_reasoning_pipeline()

    results = retriever.retrieve(question, top_k=20)
    reranked = reranker.rerank(question, results, top_k=5)
    context = compressor.compress(question, reranked)
    query_result = reasoning.query_analysis(question, context)

    return context, query_result.answer


def main():
    print("=" * 80)
    print("DOCUMENT INTELLIGENCE ENGINE - EVALUATION ON RESEARCH PAPER")
    print("=" * 80)
    print()

    print("[1/5] Parsing PDF document...")
    parsed = parse_pdf("22705iied.pdf")
    print(
        f"      Parsed: {len(parsed.sections)} sections, {len(parsed.full_text)} chars"
    )

    print("[2/5] Creating hierarchical chunks...")
    structure = chunk_document(parsed)
    print(f"      Created: {len(structure.nodes)} nodes")

    print("[3/5] Building vector store and BM25 index...")
    embedder = get_embedder()
    store = get_vector_store()
    store.add_nodes(structure.nodes)
    retriever = get_hybrid_retriever()
    retriever.rebuild_index()
    print(f"      Stored: {store.get_stats()['total_nodes']} nodes")

    print("[4/5] Running evaluation on test cases...")
    print()

    test_cases = [
        TestCase(
            question="What is this research paper about?",
            expected_key_points=["research", "study", "methodology"],
        ),
        TestCase(
            question="What are the main findings or conclusions?",
            expected_key_points=["findings", "results", "conclusions"],
        ),
        TestCase(
            question="What methodology was used in this research?",
            expected_key_points=["methodology", "approach", "method"],
        ),
        TestCase(
            question="Who are the authors of this paper?",
            expected_key_points=["author", "researcher"],
        ),
        TestCase(
            question="What are the key recommendations?",
            expected_key_points=["recommendation", "suggestion"],
        ),
    ]

    eval_pipeline = EvaluationPipeline()

    results = []
    for i, test_case in enumerate(test_cases):
        print(f"      Test {i + 1}: {test_case.question[:50]}...")

        try:
            context, answer = run_query_pipeline(test_case.question)

            eval_result = eval_pipeline.evaluate_query_response(
                question=test_case.question, context=context, generated_answer=answer
            )

            results.append(
                {
                    "question": test_case.question,
                    "answer": answer,
                    "sources": context.source_nodes[:3] if context.source_nodes else [],
                    "evaluation": eval_result.to_dict(),
                }
            )

            print(
                f"            Faithfulness: {eval_result.faithfulness_score:.2f} | "
                f"Relevance: {eval_result.relevance_score:.2f} | "
                f"Overall: {eval_result.overall_score:.2f}"
            )

        except Exception as e:
            print(f"            ERROR: {e}")
            results.append(
                {
                    "question": test_case.question,
                    "answer": "ERROR",
                    "evaluation": {"error": str(e)},
                }
            )

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    valid_results = [r for r in results if "error" not in r["evaluation"]]

    if valid_results:
        avg_faithfulness = sum(
            r["evaluation"]["faithfulness_score"] for r in valid_results
        ) / len(valid_results)
        avg_relevance = sum(
            r["evaluation"]["relevance_score"] for r in valid_results
        ) / len(valid_results)
        avg_overall = sum(
            r["evaluation"]["overall_score"] for r in valid_results
        ) / len(valid_results)
        avg_concision = sum(
            r["evaluation"].get("concision_score", 1.0) for r in valid_results
        ) / len(valid_results)
        avg_groundedness = sum(
            r["evaluation"].get("groundedness_score", 1.0) for r in valid_results
        ) / len(valid_results)
        avg_info_density = sum(
            r["evaluation"].get("information_density_score", 1.0) for r in valid_results
        ) / len(valid_results)

        print()
        print(f"Document: 22705iied.pdf (Research Paper)")
        print(f"Total Test Cases: {len(results)}")
        print(f"Valid Results: {len(valid_results)}")
        print()
        print(
            f"Average Faithfulness Score:       {avg_faithfulness:.3f} ({avg_faithfulness * 100:.1f}%)"
        )
        print(
            f"Average Relevance Score:         {avg_relevance:.3f} ({avg_relevance * 100:.1f}%)"
        )
        print(
            f"Average Concision Score:         {avg_concision:.3f} ({avg_concision * 100:.1f}%)"
        )
        print(
            f"Average Groundedness Score:      {avg_groundedness:.3f} ({avg_groundedness * 100:.1f}%)"
        )
        print(
            f"Average Info Density Score:     {avg_info_density:.3f} ({avg_info_density * 100:.1f}%)"
        )
        print(
            f"Average Overall Score:          {avg_overall:.3f} ({avg_overall * 100:.1f}%)"
        )
        print()

        if avg_overall >= 0.9:
            grade = "A - EXCELLENT (Production-grade)"
        elif avg_overall >= 0.8:
            grade = "B - GOOD"
        elif avg_overall >= 0.7:
            grade = "C - SATISFACTORY"
        elif avg_overall >= 0.6:
            grade = "D - NEEDS IMPROVEMENT"
        else:
            grade = "F - POOR"

        print(f"SYSTEM GRADE: {grade}")
        print()

        faithful_count = sum(1 for r in valid_results if r["evaluation"]["faithful"])
        relevant_count = sum(1 for r in valid_results if r["evaluation"]["relevant"])
        grounded_count = sum(
            1 for r in valid_results if r["evaluation"].get("grounded", True)
        )

        print(
            f"Faithful Responses:    {faithful_count}/{len(valid_results)} ({faithful_count / len(valid_results) * 100:.1f}%)"
        )
        print(
            f"Relevant Responses:    {relevant_count}/{len(valid_results)} ({relevant_count / len(valid_results) * 100:.1f}%)"
        )
        print(
            f"Grounded Responses:    {grounded_count}/{len(valid_results)} ({grounded_count / len(valid_results) * 100:.1f}%)"
        )

    print()
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for i, r in enumerate(results):
        print()
        print(f"[Test {i + 1}] {r['question']}")
        answer = r.get("answer", "N/A")
        if len(answer) > 300:
            print(f"Answer: {answer[:300]}...")
        else:
            print(f"Answer: {answer}")

        if "error" in r["evaluation"]:
            print(f"ERROR: {r['evaluation']['error']}")
        else:
            eval_data = r["evaluation"]
            print(
                f"  Faithful: {eval_data['faithful']} | Relevant: {eval_data['relevant']} | Grounded: {eval_data.get('grounded', True)}"
            )
            print(f"  Overall: {eval_data['overall_score']:.2f}")

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    import json
    from datetime import datetime

    output_file = f"evaluation_results_research_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "document": "22705iied.pdf (Research Paper)",
                "summary": {
                    "avg_faithfulness": avg_faithfulness if valid_results else 0,
                    "avg_relevance": avg_relevance if valid_results else 0,
                    "avg_concision": avg_concision if valid_results else 0,
                    "avg_groundedness": avg_groundedness if valid_results else 0,
                    "avg_info_density": avg_info_density if valid_results else 0,
                    "avg_overall": avg_overall if valid_results else 0,
                    "grade": grade if valid_results else "N/A",
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
