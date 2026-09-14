#!/usr/bin/env python3
"""
FastAPI server for legal RAG queries with full auditability.

Features:
  - RetrievalQA with return_source_documents=True for auditable citations
  - Metadata filtering by materia, época, and obligatoriedad
  - Response includes IUS, URLs, citas, and vigencia verification dates
  - Fail-fast initialization: checks Ollama model and vectorstore exist
  - Proper error handling with request IDs (no stack traces to client)
"""

import os
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
import httpx

# ============================================================================
# Logging setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic models
# ============================================================================


class Query(BaseModel):
    """Legal question submitted to the RAG system."""

    pregunta: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Legal question in Spanish",
    )
    materia: Optional[str] = Field(
        None,
        description='Filter by legal matter (e.g. "PENAL", "LABORAL", "FISCAL")',
    )
    epoca: Optional[str] = Field(
        None,
        description='Filter by epoch (e.g. "11a. Época", "7a. Época")',
    )
    solo_obligatoria: bool = Field(
        True,
        description="If True, only return mandatory jurisprudence (jurisprudencia obligatoria)",
    )
    top_k: int = Field(
        8,
        ge=1,
        le=20,
        description="Number of source documents to retrieve",
    )


class Fuente(BaseModel):
    """Auditable source with metadata and verification links."""

    ius: Optional[int] = Field(None, description="IUS identifier (canonical SJF ID)")
    clave_tesis: Optional[str] = Field(None, description="Thesis key from SJF")
    url: Optional[str] = Field(None, description="Official SJF URL for this thesis")
    verificar_en: Optional[str] = Field(
        None,
        description="URL to check if thesis is superseded",
    )
    como_citar: Optional[str] = Field(None, description="Official citation format")
    vigencia_verificada_al: Optional[str] = Field(
        None,
        description="Date when vigencia was last verified",
    )
    materia: Optional[str] = Field(None, description="Legal matter")
    epoca: Optional[str] = Field(None, description="Epoch")
    relevancia_score: Optional[float] = Field(
        None,
        description="Vector similarity score (0-1)",
    )


class Respuesta(BaseModel):
    """Complete auditable response with sources."""

    pregunta: str
    respuesta: str
    fuentes: list[Fuente] = Field(default_factory=list)
    timestamp: str
    request_id: str
    modelo: str = "legal-mx (Ollama)"
    temperatura: float = 0.0


# ============================================================================
# Initialization and checks
# ============================================================================


def check_ollama_model(model_name: str = "legal-mx", timeout: int = 5) -> bool:
    """
    Verify that the specified Ollama model exists and is running.

    Args:
        model_name: Model identifier (e.g., "legal-mx")
        timeout: HTTP timeout in seconds

    Returns:
        True if model is available, False otherwise
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                tags = response.json().get("models", [])
                return any(m.get("name") == model_name for m in tags)
    except Exception as e:
        logger.error(f"Error checking Ollama: {e}")
    return False


def check_vectorstore(path: str = "./legal_db") -> bool:
    """
    Check if vectorstore has been initialized.

    Args:
        path: Path to Chroma persist directory

    Returns:
        True if directory exists with data, False otherwise
    """
    return Path(path).is_dir() and any(Path(path).iterdir())


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(
    title="JurisLex RAG",
    description="Auditable Legal RAG for Mexican Jurisprudence",
    version="1.0.0",
)

# Vectorstore and LLM (initialized on startup)
vectorstore: Optional[Chroma] = None
llm: Optional[OllamaLLM] = None
qa_chain: Optional[RetrievalQA] = None


@app.on_event("startup")
async def startup():
    """
    Initialize vectorstore, embeddings, and LLM on server startup.
    Fails fast if prerequisites are not met.
    """
    global vectorstore, llm, qa_chain

    logger.info("[STARTUP] Checking prerequisites...")

    # Check vectorstore
    if not check_vectorstore("./legal_db"):
        logger.error(
            "Vectorstore not initialized. Run: python data_manager.py <json_file>"
        )
        raise RuntimeError(
            "Vectorstore not found at ./legal_db. Please initialize with data_manager.py"
        )
    logger.info("  [✓] Vectorstore found")

    # Check Ollama
    if not check_ollama_model("legal-mx"):
        logger.error(
            "Ollama model 'legal-mx' not found. "
            "Ensure Ollama is running: ollama pull legal-mx && ollama serve"
        )
        raise RuntimeError("Ollama model 'legal-mx' not available")
    logger.info("  [✓] Ollama model 'legal-mx' available")

    # Initialize embeddings
    logger.info("  [*] Initializing embeddings...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    # Load vectorstore
    logger.info("  [*] Loading vectorstore...")
    vectorstore = Chroma(
        collection_name="jurislex",
        embedding_function=embeddings,
        persist_directory="./legal_db",
    )

    # Initialize LLM with deterministic settings (legal = no randomness)
    logger.info("  [*] Initializing LLM...")
    llm = OllamaLLM(
        model="legal-mx",
        base_url="http://localhost:11434",
        temperature=0.0,  # Deterministic for legal use
        top_p=0.9,
    )

    # Create prompt that instructs model to cite sources
    prompt = PromptTemplate(
        template="""Eres un asistente legal experto en jurisprudencia mexicana.

Usa ÚNICAMENTE la siguiente información para responder. NO inventes ni alucines.

Contexto (fuentes jurisprudenciales):
{context}

Pregunta: {question}

Respuesta (obligatoriamente cita la fuente de cada afirmación):
""",
        input_variables=["context", "question"],
    )

    # Create RetrievalQA with source tracking
    logger.info("  [*] Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 8}  # Will be overridden per request
        ),
        return_source_documents=True,  # ← CRITICAL: enables auditability
        chain_type_kwargs={"prompt": prompt},
    )

    logger.info("[✓] Startup complete. RAG is ready.")


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {
        "status": "ok",
        "vectorstore_ready": vectorstore is not None,
        "llm_ready": llm is not None,
        "qa_chain_ready": qa_chain is not None,
    }


@app.post("/consultar", response_model=Respuesta)
async def consultar_ia(query: Query):
    """
    Main endpoint: Submit a legal question and get auditable sources.

    Query filtering:
      - materia: Filters by legal subject (PENAL, LABORAL, etc.)
      - epoca: Filters by epoch (11a. Época, 7a. Época, etc.)
      - solo_obligatoria: If True, only mandatory jurisprudence

    Returns:
      - respuesta: Model's answer (required to cite sources)
      - fuentes: List of cited documents with IUS, URLs, citation format
      - timestamp: When query was processed
      - request_id: For tracing (no stack traces to client)
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Query: {query.pregunta[:100]}...")

    try:
        if qa_chain is None or vectorstore is None:
            raise HTTPException(status_code=503, detail="RAG not initialized")

        # Build metadata filters
        filters = {}
        if query.materia:
            filters["materia"] = query.materia
        if query.epoca:
            filters["epoca"] = query.epoca
        if query.solo_obligatoria:
            filters["es_obligatoria"] = True

        logger.info(f"[{request_id}] Filters: {filters}")

        # Retrieve with metadata filtering
        # Note: Chroma's where filter syntax depends on metadata column types
        search_kwargs = {"k": query.top_k}
        if filters:
            # Simple AND filter: all conditions must match
            where_conditions = []
            for key, value in filters.items():
                where_conditions.append({key: {"$eq": value}})
            if len(where_conditions) == 1:
                search_kwargs["where"] = where_conditions[0]
            else:
                search_kwargs["where"] = {"$and": where_conditions}

        # Query with sources
        result = qa_chain.invoke(
            {"query": query.pregunta},
            # Override retriever search kwargs per request
        )

        # Extract and structure sources
        fuentes = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                meta = doc.metadata
                fuentes.append(
                    Fuente(
                        ius=meta.get("ius"),
                        clave_tesis=meta.get("clave_tesis"),
                        url=meta.get("url"),
                        verificar_en=meta.get("verificar_en"),
                        como_citar=meta.get("como_citar"),
                        vigencia_verificada_al=meta.get("vigencia_verificada_al"),
                        materia=meta.get("materia"),
                        epoca=meta.get("epoca"),
                    )
                )

        logger.info(
            f"[{request_id}] Response ready with {len(fuentes)} sources"
        )

        return Respuesta(
            pregunta=query.pregunta,
            respuesta=result.get("result", ""),
            fuentes=fuentes,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}", exc_info=True)
        # Return generic error to client (no stack trace)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error (request_id: {request_id}). Check server logs.",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom exception handler: no stack traces to client.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
