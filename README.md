# RegAI

**RAG over Indian regulatory documents. Ask questions about SEBI circulars, RBI guidelines, and AMFI regulations in plain English.**

Indian financial regulation is spread across thousands of SEBI circulars, RBI master directions, AMFI guidelines, and LODR amendments. Finding the right answer means searching through PDFs, cross-referencing amendments, and interpreting legalese. RegAI does this for you.

## How it works

1. **Ingest** — Parse SEBI/RBI circular PDFs with domain-specific chunking (handles cross-references, tables, amendment chains)
2. **Embed** — Generate embeddings optimized for regulatory text
3. **Retrieve** — Hybrid search (semantic + keyword) with citation tracking
4. **Generate** — Claude answers with exact circular references and quotes

## Why domain-specific chunking matters

Generic RAG chunking breaks on regulatory text because:
- Circulars reference other circulars ("as amended by SEBI/HO/IMD/DF2/CIR/P/2021/024")
- Tables contain critical numerical thresholds that lose context when split
- Amendments modify specific clauses — you need the original + amendment together
- Definitions in one section apply to rules 50 pages later

RegAI's chunking strategy preserves these relationships.

## Stack

| Layer | Tech |
|-------|------|
| LLM | Claude (via Anthropic API) |
| Embeddings | text-embedding-3-large |
| Vector DB | pgvector on Supabase |
| PDF parsing | pymupdf + unstructured |
| Framework | Python + FastAPI |
| Frontend | Next.js (optional) |

## Data sources

- SEBI circulars (sebi.gov.in) — public domain
- RBI master directions (rbi.org.in) — public domain
- AMFI guidelines (amfiindia.com) — public domain

## Status

**Work in progress.** Core RAG pipeline being built. Contributions welcome.

## License

MIT
