"""RegAI Demo - Query SEBI/RBI circulars using RAG

Usage:
    python demo.py --query "What are the KYC requirements for mutual funds?"
    python demo.py --query "SEBI circular on algorithmic trading" --top_k 5
"""

import argparse
import json
from pathlib import Path

from src.regulatory_chunker import RegulatoryChunker
from src.embeddings import get_embeddings
from src.retriever import retrieve_chunks
from src.generator import generate_answer


def load_circulars(data_dir: str = "data") -> list[dict]:
    """Load all circular JSON files from data directory."""
    circulars = []
    for f in Path(data_dir).glob("*.json"):
        with open(f) as fh:
            circulars.extend(json.load(fh))
    return circulars


def main():
    parser = argparse.ArgumentParser(description="Query regulatory circulars")
    parser.add_argument("--query", type=str, required=True, help="Your question")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to circular data")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks")
    args = parser.parse_args()

    # 1. Load and chunk circulars
    print("Loading circulars...")
    circulars = load_circulars(args.data_dir)
    chunker = RegulatoryChunker()
    chunks = []
    for circular in circulars:
        chunks.extend(chunker.chunk(circular))
    print(f"  {len(chunks)} chunks from {len(circulars)} circulars")

    # 2. Embed and retrieve
    print(f"Retrieving top {args.top_k} chunks...")
    embeddings = get_embeddings(chunks)
    results = retrieve_chunks(args.query, embeddings, top_k=args.top_k)

    if args.verbose:
        print("\n--- Retrieved Chunks ---")
        for i, r in enumerate(results):
            print(f"\n[{i+1}] (score: {r['score']:.3f})")
            print(r["text"][:300])

    # 3. Generate answer
    print("\nGenerating answer...")
    answer = generate_answer(args.query, results)
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
