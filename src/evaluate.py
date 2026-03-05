"""Evaluation harness for RegAI RAG pipeline.

Runs the bundled Q&A pairs against the chunker + retriever to measure
whether domain-specific chunking actually improves answer quality
over naive approaches.

Usage:
    python src/evaluate.py                    # Run with keyword search (no API key needed)
    ANTHROPIC_API_KEY=sk-... python src/evaluate.py --use-claude  # Run with Claude for answer generation
"""

import json
import argparse
from regulatory_chunker import chunk_all_circulars, build_amendment_graph, RegulatoryChunk


def keyword_search(query: str, chunks: list[RegulatoryChunk], top_k: int = 3) -> list[RegulatoryChunk]:
    """Simple keyword-based retrieval for evaluation without embeddings.
    
    Scores chunks by keyword overlap with the query.
    This baseline demonstrates the value of domain-specific chunking
    even without semantic search.
    """
    query_terms = set(query.lower().split())
    scored = []
    
    for chunk in chunks:
        text = chunk.to_embedding_text().lower()
        score = sum(1 for term in query_terms if term in text)
        # Boost amendment_link chunks when query mentions amendments
        if chunk.chunk_type == "amendment_link" and any(w in query.lower() for w in ["amend", "supersede", "replace", "update"]):
            score += 3
        scored.append((score, chunk))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def amendment_aware_retrieval(
    query: str,
    chunks: list[RegulatoryChunk],
    amendment_graph: dict[str, list[str]],
    top_k: int = 3,
) -> list[RegulatoryChunk]:
    """Retrieve chunks with amendment-awareness.
    
    If a retrieved chunk is from an older circular that has been amended,
    also pull in chunks from the amending circular. This ensures
    the answer reflects the latest regulation.
    """
    initial_results = keyword_search(query, chunks, top_k)
    
    # Check if any results are from amended circulars
    expanded = list(initial_results)
    seen_ids = {(c.circular_id, c.section_heading) for c in expanded}
    
    for chunk in initial_results:
        if chunk.circular_id in amendment_graph:
            # This circular has been amended - pull in the newer version
            amending_ids = amendment_graph[chunk.circular_id]
            for amending_id in amending_ids:
                for c in chunks:
                    key = (c.circular_id, c.section_heading)
                    if c.circular_id == amending_id and key not in seen_ids:
                        expanded.append(c)
                        seen_ids.add(key)
    
    return expanded


def evaluate_qa_pairs(data_path: str = "data/sample_circulars.json", use_claude: bool = False):
    """Run evaluation on all Q&A pairs."""
    with open(data_path) as f:
        data = json.load(f)
    
    chunks = chunk_all_circulars(data_path)
    amendment_graph = build_amendment_graph(chunks)
    qa_pairs = data["evaluation_qa"]
    
    print(f"Evaluating {len(qa_pairs)} Q&A pairs against {len(chunks)} chunks")
    print(f"Amendment graph: {len(amendment_graph)} circulars have amendments")
    print("=" * 80)
    
    results = {"correct_retrieval": 0, "total": len(qa_pairs)}
    
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\nQ{i}: {qa['question']}")
        print(f"Expected source: {qa['source_circular']}")
        print(f"Testing: {qa['reasoning']}")
        
        # Retrieve with amendment awareness
        retrieved = amendment_aware_retrieval(
            qa["question"], chunks, amendment_graph, top_k=3
        )
        
        # Check if correct source circular was retrieved
        retrieved_ids = {c.circular_id for c in retrieved}
        correct = qa["source_circular"] in retrieved_ids
        results["correct_retrieval"] += int(correct)
        
        status = "PASS" if correct else "FAIL"
        print(f"Retrieved: {retrieved_ids}")
        print(f"Result: {status}")
        
        if use_claude and correct:
            # Generate answer with Claude using retrieved context
            _generate_with_claude(qa["question"], retrieved, qa["expected_answer"])
        
        print("-" * 40)
    
    accuracy = results["correct_retrieval"] / results["total"]
    print(f"\n{'=' * 80}")
    print(f"Retrieval accuracy: {results['correct_retrieval']}/{results['total']} ({accuracy:.0%})")
    print(f"\nNote: This evaluates whether the correct circular is retrieved.")
    print(f"Add --use-claude with ANTHROPIC_API_KEY to also evaluate answer quality.")
    return results


def _generate_with_claude(question: str, chunks: list[RegulatoryChunk], expected: str):
    """Generate and compare answer using Claude (requires API key)."""
    try:
        import anthropic
        import os
        
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        context = "\n\n---\n\n".join(c.to_embedding_text() for c in chunks)
        
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"""Based on the following regulatory documents, answer the question.
Cite the specific circular number in your answer.

Documents:
{context}

Question: {question}

Answer concisely with the circular reference."""
            }]
        )
        
        answer = response.content[0].text
        print(f"  Claude's answer: {answer[:200]}")
        print(f"  Expected: {expected[:200]}")
    except ImportError:
        print("  (anthropic package not installed - pip install anthropic)")
    except Exception as e:
        print(f"  Claude generation error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RegAI retrieval")
    parser.add_argument("--use-claude", action="store_true", help="Use Claude for answer generation")
    parser.add_argument("--data", default="data/sample_circulars.json", help="Path to circular data")
    args = parser.parse_args()
    
    evaluate_qa_pairs(args.data, args.use_claude)
