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

    # Retrieve
    results = retriever.retrieve(question, top_k=20)

    # Rerank
    reranked = reranker.rerank(question, results, top_k=5)

    # Compress
    context = compressor.compress(question, reranked)

    # Generate answer
    query_result = reasoning.query_analysis(question, context)

    return context, query_result.answer


def main():
    print("=" * 80)
    print("DOCUMENT INTELLIGENCE ENGINE - EVALUATION PIPELINE")
    print("=" * 80)
    print()

    # Parse the PDF
    print("[1/5] Parsing PDF document...")
    parsed = parse_pdf("DICKSON SARPONG_AI Engineer.pdf")
    print(f"      Parsed: {len(parsed.sections)} sections")

    # Chunk
    print("[2/5] Creating hierarchical chunks...")
    structure = chunk_document(parsed)
    print(f"      Created: {len(structure.nodes)} nodes")

    # Store
    print("[3/5] Building vector store and BM25 index...")
    embedder = get_embedder()
    store = get_vector_store()
    store.add_nodes(structure.nodes)

    # Rebuild BM25
    retriever = get_hybrid_retriever()
    retriever.rebuild_index()
    print(f"      Stored: {store.get_stats()['total_nodes']} nodes")

    # Define test cases based on PDF content
    print("[4/5] Running evaluation on test cases...")
    print()

    test_cases = [
        TestCase(
            question="Who is Dickson Sarpong and what is his experience?",
            expected_key_points=[
                "AI Engineer",
                "LLM interaction",
                "repository analysis",
            ],
        ),
        TestCase(
            question="What technical skills does Dickson have?",
            expected_key_points=["Python", "LLMs", "safety mechanisms"],
        ),
        TestCase(
            question="What projects has Dickson worked on?",
            expected_key_points=[
                "automated repository analysis",
                "governance",
                "safety",
            ],
        ),
        TestCase(
            question="What is the document about overall?",
            expected_key_points=["resume", "AI Engineer", "experience"],
        ),
        TestCase(
            question="What programming languages is Dickson proficient in?",
            expected_key_points=["Python", "programming languages"],
        ),
    ]

    # Run evaluation
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
                    "evaluation": eval_result.to_dict(),
                }
            )

            print(
                f"            Faithfulness: {eval_result.faithfulness_score:.2f} | "
                f"Relevance: {eval_result.relevance_score:.2f} | "
                f"Overall: {eval_result.overall_score:.2f}"
            )

            if eval_result.hallucinations:
                print(f"            Hallucinations: {eval_result.hallucinations}")

        except Exception as e:
            import traceback

            print(f"            ERROR: {type(e).__name__}: {e}")
            print(f"            Trace: {traceback.format_exc()[:500]}")
            results.append(
                {
                    "question": test_case.question,
                    "answer": "ERROR",
                    "evaluation": {"error": f"{type(e).__name__}: {str(e)}"},
                }
            )

    # Aggregate results
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
            f"Average Info Density Score:      {avg_info_density:.3f} ({avg_info_density * 100:.1f}%)"
        )
        print(
            f"Average Overall Score:           {avg_overall:.3f} ({avg_overall * 100:.1f}%)"
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
        specific_count = sum(
            1 for r in valid_results if r["evaluation"].get("specific", True)
        )

        print(
            f"Faithful Responses:    {faithful_count}/{len(valid_results)} ({faithful_count / len(valid_results) * 100:.1f}%)"
        )
        print(
            f"Relevant Responses:     {relevant_count}/{len(valid_results)} ({relevant_count / len(valid_results) * 100:.1f}%)"
        )
        print(
            f"Grounded Responses:    {grounded_count}/{len(valid_results)} ({grounded_count / len(valid_results) * 100:.1f}%)"
        )
        print(
            f"Specific Responses:    {specific_count}/{len(valid_results)} ({specific_count / len(valid_results) * 100:.1f}%)"
        )

        all_hallucinations = []
        all_missing = []
        all_issues = []

        for r in valid_results:
            all_hallucinations.extend(r["evaluation"].get("hallucinations", []))
            all_missing.extend(r["evaluation"].get("missing_info", []))
            all_issues.extend(r["evaluation"].get("issues", []))

        print()
        print(
            "Hallucinations:", all_hallucinations[:3] if all_hallucinations else "None"
        )
        print("Missing Info:", all_missing[:3] if all_missing else "None")

    print()
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for i, r in enumerate(results):
        print()
        print(f"[Test {i + 1}] {r['question']}")
        answer = r.get("answer", "N/A")
        if len(answer) > 250:
            print(f"Answer: {answer[:250]}...")
        else:
            print(f"Answer: {answer}")
        sources = r.get("sources", [])
        if sources:
            print(f"Evidence: {sources[:3]}")

        if "error" in r["evaluation"]:
            print(f"ERROR: {r['evaluation']['error']}")
        else:
            eval_data = r["evaluation"]
            print(
                f"  Faithful: {eval_data['faithful']} | Relevant: {eval_data['relevant']} | Grounded: {eval_data.get('grounded', True)} | Specific: {eval_data.get('specific', True)}"
            )
            print(
                f"  Scores - Faith:{eval_data['faithfulness_score']:.2f} | Rel:{eval_data['relevance_score']:.2f} | Conc:{eval_data.get('concision_score', 1.0):.2f} | Ground:{eval_data.get('groundedness_score', 1.0):.2f} | Info:{eval_data.get('information_density_score', 1.0):.2f} | Overall:{eval_data['overall_score']:.2f}"
            )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    # Save results to file
    import json
    from datetime import datetime

    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "document": "DICKSON SARPONG_AI Engineer.pdf",
                "summary": {
                    "avg_faithfulness": avg_faithfulness if valid_results else 0,
                    "avg_relevance": avg_relevance if valid_results else 0,
                    "avg_concision": avg_concision if valid_results else 0,
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
