"""Domain-specific chunking for Indian financial regulatory documents.

Handles the unique challenges of SEBI/RBI circulars:
- Table preservation (keeps rows with headers for context)
- Amendment chain linking (connects amending circulars to originals)
- Cross-reference resolution (inline circular references)
- Section-aware splitting (respects regulatory document structure)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegulatoryChunk:
    """A chunk of regulatory text with full metadata for retrieval."""
    content: str
    circular_id: str
    circular_title: str
    section_heading: str
    chunk_type: str  # 'text', 'table', 'amendment_link'
    issuing_authority: str  # 'SEBI' or 'RBI'
    date: str
    amends: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)
    table_headers: Optional[list[str]] = None

    def to_embedding_text(self) -> str:
        """Format chunk for embedding generation.
        
        Prepends metadata so the embedding captures regulatory context,
        not just raw text.
        """
        parts = [
            f"[{self.issuing_authority}] {self.circular_id}",
            f"Title: {self.circular_title}",
            f"Section: {self.section_heading}",
            f"Date: {self.date}",
        ]
        if self.amends:
            parts.append(f"Amends: {', '.join(self.amends)}")
        parts.append(f"\n{self.content}")
        return "\n".join(parts)


def extract_cross_references(text: str) -> list[str]:
    """Extract SEBI/RBI circular reference numbers from text.
    
    Regulatory text frequently references other circulars inline.
    These references are critical for building the amendment graph.
    
    Patterns matched:
    - SEBI/HO/IMD/DF3/CIR/P/2017/114
    - RBI/2023-24/45/FIDD.CO.Plan.BC.5/04.09.01/2023-24
    """
    sebi_pattern = r'SEBI/[A-Z/]+/\d{4}/\d+'
    rbi_pattern = r'RBI/\d{4}-\d{2}/\d+[\w./]*'
    
    refs = re.findall(sebi_pattern, text)
    refs.extend(re.findall(rbi_pattern, text))
    return list(set(refs))


def format_table_as_text(table_data: dict) -> str:
    """Convert structured table to text preserving row-header relationships.
    
    Naive chunking splits table rows from headers, losing context.
    This formatter keeps each row paired with its column headers
    so the embedding captures the full meaning.
    
    Example output:
        Category: Equity | Sub-Category: Large Cap | Min Equity %: 80 | Benchmark: NIFTY 100 TRI
    """
    headers = table_data["headers"]
    rows = table_data["rows"]
    formatted_rows = []
    
    for row in rows:
        pairs = [f"{h}: {v}" for h, v in zip(headers, row)]
        formatted_rows.append(" | ".join(pairs))
    
    return "\n".join(formatted_rows)


def chunk_circular(circular: dict) -> list[RegulatoryChunk]:
    """Chunk a single circular into retrieval-ready pieces.
    
    Strategy:
    - Text sections: Keep as single chunks (regulatory sections are
      typically 1-3 paragraphs and shouldn't be split mid-clause)
    - Table sections: Format with headers preserved per row
    - Always attach circular metadata for citation tracking
    - Extract cross-references for amendment graph building
    """
    chunks = []
    
    for section in circular["sections"]:
        if section["type"] == "text":
            content = section["content"]
            cross_refs = extract_cross_references(content)
        elif section["type"] == "table":
            content = format_table_as_text(section["content"])
            cross_refs = []
        else:
            continue
        
        chunk = RegulatoryChunk(
            content=content,
            circular_id=circular["id"],
            circular_title=circular["title"],
            section_heading=section["heading"],
            chunk_type=section["type"],
            issuing_authority=circular["issuing_authority"],
            date=circular["date"],
            amends=circular.get("amends", []),
            cross_references=cross_refs,
            table_headers=section["content"].get("headers") if section["type"] == "table" else None,
        )
        chunks.append(chunk)
    
    # Add amendment link chunks for explicit connection
    if circular.get("amends"):
        amendment_text = (
            f"This circular ({circular['id']}) amends the following: "
            f"{', '.join(circular['amends'])}. "
            f"When answering questions about topics covered in this circular, "
            f"this version supersedes the earlier circulars."
        )
        chunks.append(RegulatoryChunk(
            content=amendment_text,
            circular_id=circular["id"],
            circular_title=circular["title"],
            section_heading="Amendment Chain",
            chunk_type="amendment_link",
            issuing_authority=circular["issuing_authority"],
            date=circular["date"],
            amends=circular["amends"],
            cross_references=circular["amends"],
        ))
    
    return chunks


def chunk_all_circulars(data_path: str = "data/sample_circulars.json") -> list[RegulatoryChunk]:
    """Load and chunk all circulars from the sample data file."""
    with open(data_path) as f:
        data = json.load(f)
    
    all_chunks = []
    for circular in data["circulars"]:
        all_chunks.extend(chunk_circular(circular))
    
    return all_chunks


def build_amendment_graph(chunks: list[RegulatoryChunk]) -> dict[str, list[str]]:
    """Build a graph of which circulars amend which.
    
    Returns a dict mapping circular_id -> list of circular_ids that amend it.
    This is used during retrieval to pull in related circulars
    when a query matches an older circular that has been amended.
    """
    graph = {}
    for chunk in chunks:
        if chunk.chunk_type == "amendment_link":
            for amended_id in chunk.amends:
                if amended_id not in graph:
                    graph[amended_id] = []
                graph[amended_id].append(chunk.circular_id)
    return graph


if __name__ == "__main__":
    # Demo: chunk the sample circulars and show results
    chunks = chunk_all_circulars()
    print(f"Total chunks: {len(chunks)}")
    print(f"\nChunk types: {[c.chunk_type for c in chunks]}")
    print(f"\nAmendment graph:")
    graph = build_amendment_graph(chunks)
    for original, amenders in graph.items():
        print(f"  {original} <- amended by {amenders}")
    print(f"\n--- Sample chunk (table with headers preserved) ---")
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    if table_chunks:
        print(table_chunks[0].to_embedding_text()[:500])
