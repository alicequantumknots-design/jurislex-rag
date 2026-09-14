# JurisLex RAG

**Auditable Legal RAG System for Mexican Jurisprudence**

A production-ready Retrieval-Augmented Generation (RAG) system for querying Mexican legal jurisprudence with full auditability, metadata filtering, and source citation.

## Overview

JurisLex RAG addresses critical gaps in naive RAG implementations for legal use:

1. **True Auditability**: Every response includes canonical IUS identifiers, official SJF URLs, citation formats, and vigencia verification dates.
2. **Metadata Filtering**: Query results filtered by legal matter (materia), epoch (época), and mandatory jurisprudence status.
3. **Semantic Chunking**: Documents split by paragraph with metadata preservation, not by arbitrary token count.
4. **Deterministic LLM**: Temperature 0.0 for legal consistency.
5. **Fail-Fast Startup**: Validates Ollama model and vectorstore before accepting queries.
6. **Batch Ingestion**: Handles 300k+ documents without OOM.

## Architecture

```
┌─────────────────────────────────────────┐
│   Corpusiuris JSON (300k+ tesis)        │
└──────────────┬──────────────────────────┘
               │
        ▼
┌─────────────────────────────────────────┐
│  data_manager.py                        │
│  ├─ Parse metadata (IUS, materia, época)│
│  ├─ Chunk by paragraph (800 chars)      │
│  └─ Embed (nomic-embed-text via Ollama) │
└──────────────┬──────────────────────────┘
               │
        ▼
┌─────────────────────────────────────────┐
│  Chroma Vectorstore (./legal_db)        │
│  ├─ 300k+ documents × chunks per doc    │
│  └─ Metadata indexed for filtering      │
└──────────────┬──────────────────────────┘
               │
        ▼
┌─────────────────────────────────────────┐
│  app.py (FastAPI)                       │
│  ├─ Metadata filter (materia, época)    │
│  ├─ RetrievalQA with sources            │
│  ├─ Ollama LLM (legal-mx, temp=0.0)     │
│  └─ Response: citation + fuentes + IUS  │
└──────────────┬──────────────────────────┘
               │
        ▼
┌─────────────────────────────────────────┐
│  Client                                 │
│  ├─ pregunta (question)                 │
│  ├─ respuesta (answer + citations)      │
│  ├─ fuentes (sources with URLs + IUS)   │
│  └─ request_id (tracing)                │
└─────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- Ollama (running locally)
- 16GB+ RAM (for batch embedding)

### Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/alicequantumknots-design/jurislex-rag.git
   cd jurislex-rag
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Ollama (in another terminal):
   ```bash
   # Pull the legal model (if not already present)
   ollama pull legal-mx
   ollama pull nomic-embed-text
   
   # Start the Ollama server
   ollama serve
   ```

## Usage

### 1. Ingest Data

Download Corpusiuris JSON and run the ingestion:

```bash
python data_manager.py /path/to/corpusiuris.json ./legal_db
```

**Output**:
- Parses 300k+ items
- Chunks by paragraph (800 char chunks, 150 overlap)
- Embeds via `nomic-embed-text` (Ollama)
- Stores in Chroma at `./legal_db`
- ~2-3 hours for full dataset on CPU

### 2. Start API Server

```bash
python app.py
```

Server starts at `http://localhost:8000`

**Checks on startup**:
- ✓ Vectorstore exists at `./legal_db`
- ✓ Ollama model `legal-mx` available
- ✓ Ollama model `nomic-embed-text` available

### 3. Query the RAG

#### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "vectorstore_ready": true,
  "llm_ready": true,
  "qa_chain_ready": true
}
```

#### Legal Query

```bash
curl -X POST http://localhost:8000/consultar \
  -H "Content-Type: application/json" \
  -d '{
    "pregunta": "¿Cuáles son los requisitos para un cateo según el artículo 16 de la Constitución?",
    "materia": "PENAL",
    "epoca": "11a. Época",
    "solo_obligatoria": true,
    "top_k": 8
  }'
```

#### Response

```json
{
  "pregunta": "¿Cuáles son los requisitos para un cateo según el artículo 16 de la Constitución?",
  "respuesta": "Los requisitos para un cateo son: [cites sources]...",
  "fuentes": [
    {
      "ius": 2026661,
      "clave_tesis": "1a./J. 44/2023 (11a.)",
      "url": "https://sjf2.scjn.gob.mx/detalle/tesis/2026661",
      "verificar_en": "https://sjf2.scjn.gob.mx/verificar/2026661",
      "como_citar": "Jurisprudencia OBLIGATORIA...",
      "vigencia_verificada_al": "2026-09-14",
      "materia": "PENAL",
      "epoca": "11a. Época",
      "relevancia_score": 0.92
    }
  ],
  "timestamp": "2026-09-14T15:05:07.123456",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "modelo": "legal-mx (Ollama)",
  "temperatura": 0.0
}
```

## Query Filters

### `materia` (Legal Subject)

- `PENAL` – Criminal law
- `LABORAL` – Labor law
- `FISCAL` – Tax law
- `CIVIL` – Civil law
- etc. (see Corpusiuris for full list)

### `epoca` (Epoch)

- `11a. Época` – Current (2011–present)
- `7a. Época` – Older case law
- etc.

### `solo_obligatoria` (Mandatory Only)

- `true` – Return only jurisprudencia obligatoria (binding precedent)
- `false` – Return all tesis (including isolated rulings)

### `top_k` (Number of Sources)

- Default: 8
- Range: 1–20

## Testing

Run the ingestion and retrieval tests:

```bash
pip install -r requirements-dev.txt
pytest test_ingestion.py -v
```

**Tests cover**:
- Metadata parsing from Corpusiuris JSON
- Chunking with metadata preservation
- Source document filtering

## Critical Design Decisions

### 1. Why Chunking?

A single tesis often has 5–15 paragraphs with independent legal arguments. Without chunking:
- A query about "requisitos de cateo" might not match a tesis with 40 paragraphs where only one discusses this.
- Chunk size 800 characters balances context (full considerando) with precision.

### 2. Why `return_source_documents=True`?

Naive RAG returns a string response. Legal RAG must return:
- IUS (official ID for verification)
- URL (to check latest status)
- Citation format ("Jurisprudencia obligatoria...")
- Vigencia date (when last verified against SJF)

Without this, responses are **not auditable**.

### 3. Why Metadata Filters?

Criminal law in the "11a. Época" differs from "7a. Época." A query about PENAL law should not match FISCAL tesis. Filters are not optional; they are essential correctness.

### 4. Why Temperature 0.0?

Legal responses must be deterministic. With temperature > 0, the same query gives different citations each run, making auditing impossible.

### 5. Why Batch Ingestion?

300k tesis × 3 chunks per tesis ≈ 900k documents. Embedding all at once:
- RAM: ~200GB
- Time: Blocks indefinitely

Batch size 500 keeps memory footprint ~2GB per batch.

## Production Deployment

### Docker

Add `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data_manager.py app.py test_ingestion.py ./
COPY legal_db ./legal_db

EXPOSE 8000
CMD ["python", "app.py"]
```

### Environment Variables

Add `.env`:

```
OLLAMA_BASE_URL=http://ollama:11434
VECTORSTORE_PATH=./legal_db
LOG_LEVEL=info
```

### Monitoring

- `/health` endpoint for liveness checks
- Request IDs in logs for tracing
- `request_id` in response JSON for correlation

## Limitations

1. **Vectorstore updates**: Ingestion is one-time. To add new tesis, re-ingest from scratch.
2. **Ollama CPU**: Embedding on CPU is slow (~2-3h for 300k tesis). GPU acceleration recommended.
3. **Retriever confidence**: Top-k retrieval may miss relevant tesis. Hybrid search (BM25 + vector) or reranking not yet implemented.
4. **Citation hallucination**: Even with temperature 0, LLM might invent citations. Prompt engineering and few-shot examples needed.

## Roadmap

- [ ] Incremental ingestion (add new tesis without full re-ingest)
- [ ] Hybrid search (BM25 + vector similarity)
- [ ] Cross-encoder reranker (improve top-k precision)
- [ ] Query expansion ("cateo" → "diligencia de inspección")
- [ ] Citation validation (LLM output vs. actual SJF)
- [ ] Fallback to SJF HTML scraping (API alternative)

## References

- [Corpusiuris API](https://www.scjn.gob.mx/) – Mexican Supreme Court jurisprudence
- [SJF2 Tesis Lookup](https://sjf2.scjn.gob.mx/) – Verification and citation
- [LangChain Docs](https://python.langchain.com) – RAG framework
- [Chroma Docs](https://docs.trychroma.com) – Vector database
- [Ollama](https://ollama.ai/) – Local LLM inference

## License

MIT

## Authors

Alice Quantum Knots Design
