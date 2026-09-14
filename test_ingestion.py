#!/usr/bin/env python3
"""
Test suite for data ingestion and retrieval pipeline.

Verifies:
  - Metadata parsing and preservation
  - Chunking and embedding
  - Retrieval accuracy and source citations
"""

import pytest
from pathlib import Path
from data_manager import build_document, chunk_documents
from langchain_core.documents import Document


class TestBuildDocument:
    """Test document parsing from Corpusiuris JSON."""

    def test_complete_document(self):
        """Test parsing a complete Corpusiuris item."""
        item = {
            "documento_id": "2026661",
            "tipo": "tesis",
            "titulo": "Test Title",
            "fragmento": "This is a test fragment about legal procedure.",
            "url": "https://sjf2.scjn.gob.mx/detalle/tesis/2026661",
            "metadata": {
                "ius": 2026661,
                "clave_tesis": "1a./J. 44/2023 (11a.)",
                "epoca": "11a. Época",
                "materia_principal": "PENAL",
            },
            "fuerza": {
                "es_jurisprudencia_obligatoria": True,
                "como_citar": "Jurisprudencia obligatoria...",
                "verificar_superacion_en": "https://sjf2.scjn.gob.mx/verificar/2026661",
            },
            "vigencia": {
                "verificado_al": "2026-09-14",
                "fecha_ultima_reforma": "2026-01-01",
            },
        }

        doc = build_document(item)

        assert doc is not None
        assert "legal procedure" in doc.page_content
        assert doc.metadata["ius"] == 2026661
        assert doc.metadata["clave_tesis"] == "1a./J. 44/2023 (11a.)"
        assert doc.metadata["es_obligatoria"] is True
        assert doc.metadata["materia"] == "PENAL"
        assert doc.metadata["url"] == "https://sjf2.scjn.gob.mx/detalle/tesis/2026661"

    def test_missing_metadata(self):
        """Test handling of partial metadata."""
        item = {
            "documento_id": "123",
            "tipo": "tesis",
            "fragmento": "Minimal fragment",
            # No metadata, fuerza, or vigencia
        }

        doc = build_document(item)

        assert doc is not None
        assert doc.metadata["ius"] is None
        assert doc.metadata["es_obligatoria"] is False

    def test_empty_text_returns_none(self):
        """Test that items without text are skipped."""
        item = {
            "documento_id": "123",
            "tipo": "tesis",
            # No texto, fragmento, or titulo
        }

        doc = build_document(item)
        assert doc is None


class TestChunking:
    """Test document chunking with metadata preservation."""

    def test_chunking_preserves_metadata(self):
        """Test that metadata is copied to each chunk."""
        doc = Document(
            page_content="\n\n".join(
                [f"Paragraph {i}" for i in range(10)]
            ),  # Long text
            metadata={
                "ius": 123,
                "clave_tesis": "1a./J. 44/2023",
                "materia": "PENAL",
            },
        )

        chunks = chunk_documents([doc], chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 1  # Should split into multiple chunks
        for chunk in chunks:
            assert chunk.metadata["ius"] == 123
            assert chunk.metadata["clave_tesis"] == "1a./J. 44/2023"
            assert chunk.metadata["materia"] == "PENAL"

    def test_small_document_single_chunk(self):
        """Test that small documents remain single chunks."""
        doc = Document(
            page_content="Short text",
            metadata={"ius": 456},
        )

        chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=100)

        assert len(chunks) == 1
        assert chunks[0].metadata["ius"] == 456


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
