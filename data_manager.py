#!/usr/bin/env python3
"""
Data ingestion module for Corpusiuris legal documents.

Handles:
  - Loading JSON from Corpusiuris API or local files
  - Parsing jurisprudence metadata (IUS, fuerza, vigencia, materia, época)
  - Chunking documents with metadata preservation
  - Embedding and storing in local Chroma vectorstore
  - Batch processing to prevent OOM on 300k+ documents
"""

import json
import os
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def build_document(item: dict) -> Optional[Document]:
    """
    Convert a Corpusiuris item to a LangChain Document with complete metadata.

    Args:
        item: Single item from Corpusiuris JSON payload

    Returns:
        Document with page_content and auditable metadata, or None if invalid
    """
    try:
        # Extract nested structures with safe defaults
        metadata = item.get("metadata") or {}
        fuerza = item.get("fuerza") or {}
        vigencia = item.get("vigencia") or {}

        # Text: prefer fragmento (excerpt) or full text; fallback to título
        texto = item.get("fragmento") or item.get("texto") or item.get("titulo", "")
        if not texto:
            return None

        return Document(
            page_content=texto,
            metadata={
                # Canonical identifiers
                "documento_id": str(item.get("documento_id", "")),
                "ius": metadata.get("ius"),
                "clave_tesis": metadata.get("clave_tesis"),
                "tipo": item.get("tipo", "tesis"),
                # Legal filters: materia and época are CRITICAL for correct retrieval
                "materia": metadata.get("materia_principal"),
                "epoca": metadata.get("epoca"),
                # Mandatory jurisprudence flag
                "es_obligatoria": bool(
                    fuerza.get("es_jurisprudencia_obligatoria")
                ),
                # Auditability: sources and verification URLs
                "url": item.get("url", ""),
                "verificar_en": fuerza.get("verificar_superacion_en", ""),
                "como_citar": fuerza.get("como_citar", ""),
                # Vigencia dates for cache invalidation
                "vigencia_verificada_al": vigencia.get("verificado_al", ""),
                "fecha_ultima_reforma": vigencia.get("fecha_ultima_reforma", ""),
            },
        )
    except Exception as e:
        print(f"  [WARN] Skipping malformed item: {e}")
        return None


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split documents into chunks while preserving metadata.

    Legal texts often have independent considerandos in different paragraphs.
    Without chunking, an entire tesis (5-15 paragraphs) becomes one vector,
    reducing retrieval precision.

    Args:
        docs: List of Document objects
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks to preserve context

    Returns:
        List of chunked documents with original metadata copied to each chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunked = []
    for doc in docs:
        splits = splitter.split_documents([doc])
        # Ensure metadata is preserved on each chunk
        for split in splits:
            # Merge original metadata into chunk
            split.metadata.update(doc.metadata)
        chunked.extend(splits)
    
    return chunked


def ingest_data(
    json_file_path: str,
    batch_size: int = 500,
    vectorstore_path: str = "./legal_db",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> Chroma:
    """
    Load, chunk, embed, and store legal documents in Chroma.

    Handles:
      - Loading from Corpusiuris JSON (flat list or {"resultados": [...]})
      - Batch processing to avoid OOM on 300k+ documents
      - Metadata preservation and auditable citation links

    Args:
        json_file_path: Path to Corpusiuris JSON file
        batch_size: Number of documents to embed per batch (memory-dependent)
        vectorstore_path: Path to persist Chroma database
        chunk_size: Character size of document chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Initialized Chroma vectorstore ready for queries
    """
    print(f"[*] Loading JSON from {json_file_path}...")
    with open(json_file_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both flat list and {"resultados": [...]} formats
    items = data.get("resultados", data) if isinstance(data, dict) else data

    print(f"[*] Parsing {len(items)} items...")
    docs = [build_document(item) for item in items]
    docs = [d for d in docs if d is not None]  # Filter out None (invalid items)
    print(f"    Valid documents: {len(docs)}")

    print(f"[*] Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    chunked_docs = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"    Total chunks: {len(chunked_docs)}")

    print(f"[*] Initializing embeddings (nomic-embed-text via Ollama)...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    print(f"[*] Creating Chroma vectorstore at {vectorstore_path}...")
    vectorstore = Chroma(
        collection_name="jurislex",
        embedding_function=embeddings,
        persist_directory=vectorstore_path,
    )

    print(f"[*] Ingesting in batches of {batch_size}...")
    for i in range(0, len(chunked_docs), batch_size):
        batch = chunked_docs[i : i + batch_size]
        vectorstore.add_documents(batch)
        progress = min(i + batch_size, len(chunked_docs))
        print(f"    [{progress}/{len(chunked_docs)}]")

    print("[✓] Ingestion complete.")
    return vectorstore


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_manager.py <json_file> [vectorstore_path]")
        sys.exit(1)

    json_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else "./legal_db"

    if not os.path.isfile(json_path):
        print(f"[ERROR] File not found: {json_path}")
        sys.exit(1)

    ingest_data(json_path, vectorstore_path=db_path)
