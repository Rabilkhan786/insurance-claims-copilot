"""Single validated configuration entry point for the application."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


REQUIRED_SECTIONS = (
    "project",
    "models",
    "pinecone",
    "retrieval",
    "chunking",
    "api",
    "table_extraction",
)


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    artifacts_dir: Path
    embedding_model: str
    cross_encoder_model: str
    llm_model: str
    llm_temperature: float
    dense_index_name: str
    sparse_index_name: str
    namespace: str
    pinecone_cloud: str
    pinecone_region: str
    dense_dimension: int
    dense_metric: str
    sparse_model: str
    dense_top_k: int
    sparse_top_k: int
    final_top_k: int
    rrf_k: int
    api_host: str
    api_port: int
    batch_size: int
    rag_cache_ttl_seconds: int
    crm_db_path: Path
    chunk_strategy: str
    max_chunk_size: int
    min_chunk_size: int
    tag_topics: bool
    chunk_skip_patterns: tuple[str, ...]
    table_extraction_enabled: bool
    table_min_rows: int
    table_min_cols: int
    pinecone_api_key: str | None
    groq_api_key: str | None


def _required(mapping: dict[str, Any], key: str) -> Any:
    """Return mapping[key] or raise a clear error if it is missing."""
    if key not in mapping:
        raise ValueError(f"Missing configuration value: {key}")
    return mapping[key]


def _load_yaml_config(root: Path) -> dict[str, Any]:
    """Read config.yaml and confirm every required section is present."""
    config_path = root / "config" / "config.yaml"
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    for section in REQUIRED_SECTIONS:
        _required(raw, section)
    return raw


def _project_kwargs(root: Path, project: dict) -> dict:
    """Build Settings fields for root/data/artifact paths."""
    return {
        "root_dir": root,
        "data_dir": root / project["data_dir"],
        "artifacts_dir": root / project["artifacts_dir"],
    }


def _model_kwargs(models: dict) -> dict:
    """Build Settings fields for embedding, reranker, and LLM model names."""
    return {
        "embedding_model": models["embedding"],
        "cross_encoder_model": models["cross_encoder"],
        "llm_model": models["llm"],
        "llm_temperature": float(models["temperature"]),
    }


def _pinecone_kwargs(pinecone: dict) -> dict:
    """Build Settings fields for Pinecone index configuration."""
    return {
        "dense_index_name": pinecone["dense_index"],
        "sparse_index_name": pinecone["sparse_index"],
        "namespace": pinecone["namespace"],
        "pinecone_cloud": pinecone["cloud"],
        "pinecone_region": pinecone["region"],
        "dense_dimension": int(pinecone["dense_dimension"]),
        "dense_metric": pinecone["dense_metric"],
        "sparse_model": pinecone["sparse_model"],
    }


def _retrieval_kwargs(retrieval: dict) -> dict:
    """Build Settings fields for hybrid retrieval top-k and RRF parameters."""
    return {
        "dense_top_k": int(retrieval["dense_top_k"]),
        "sparse_top_k": int(retrieval["sparse_top_k"]),
        "final_top_k": int(retrieval["final_top_k"]),
        "rrf_k": int(retrieval["rrf_k"]),
    }


def _chunking_kwargs(chunking: dict) -> dict:
    """Build Settings fields for text chunking strategy and size limits."""
    return {
        "chunk_strategy": chunking["strategy"],
        "max_chunk_size": int(chunking["max_chunk_size"]),
        "min_chunk_size": int(chunking["min_chunk_size"]),
        "tag_topics": bool(chunking["tag_topics"]),
        "chunk_skip_patterns": tuple(chunking["skip_patterns"]),
    }


def _table_kwargs(table_extraction: dict) -> dict:
    """Build Settings fields for PDF table extraction rules."""
    return {
        "table_extraction_enabled": bool(table_extraction["enabled"]),
        "table_min_rows": int(table_extraction["min_rows"]),
        "table_min_cols": int(table_extraction["min_cols"]),
    }


def _infra_kwargs(root: Path, raw: dict) -> dict:
    """Build Settings fields for the API server and database paths."""
    api = raw["api"]
    database = raw.get("database", {})
    cache = raw.get("cache", {})
    return {
        "api_host": api["host"],
        "api_port": int(api["port"]),
        "batch_size": int(raw["batch_size"]),
        "rag_cache_ttl_seconds": int(cache.get("rag_ttl_seconds", 3600)),
        "crm_db_path": root / database.get("crm_path", "data/crm.db"),
    }


def _secret_kwargs() -> dict:
    """Build Settings fields for API keys, read from the environment only."""
    return {
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),
    }


def load_settings() -> Settings:
    """Load env vars and config.yaml into a validated Settings object."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    raw = _load_yaml_config(root)
    return Settings(
        **_project_kwargs(root, raw["project"]),
        **_model_kwargs(raw["models"]),
        **_pinecone_kwargs(raw["pinecone"]),
        **_retrieval_kwargs(raw["retrieval"]),
        **_chunking_kwargs(raw["chunking"]),
        **_table_kwargs(raw["table_extraction"]),
        **_infra_kwargs(root, raw),
        **_secret_kwargs(),
    )


settings = load_settings()
