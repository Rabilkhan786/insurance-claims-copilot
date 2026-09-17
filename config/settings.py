"""Load application settings from YAML and environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application configuration used across the project."""

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
    max_chunk_size: int
    min_chunk_size: int
    tag_topics: bool
    chunk_skip_patterns: tuple[str, ...]
    table_extraction_enabled: bool
    table_min_rows: int
    table_min_cols: int
    pinecone_api_key: str | None
    groq_api_key: str | None


def load_settings() -> Settings:
    """Read config.yaml and build the project settings object."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    config_path = root / "config" / "config.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    required_sections = (
        "project",
        "models",
        "pinecone",
        "retrieval",
        "chunking",
        "api",
        "table_extraction",
    )
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    project = config["project"]
    models = config["models"]
    pinecone = config["pinecone"]
    retrieval = config["retrieval"]
    chunking = config["chunking"]
    api = config["api"]
    table_extraction = config["table_extraction"]
    database = config.get("database", {})
    cache = config.get("cache", {})

    return Settings(
        root_dir=root,
        data_dir=root / project["data_dir"],
        artifacts_dir=root / project["artifacts_dir"],
        embedding_model=models["embedding"],
        cross_encoder_model=models["cross_encoder"],
        llm_model=models["llm"],
        llm_temperature=float(models["temperature"]),
        dense_index_name=pinecone["dense_index"],
        sparse_index_name=pinecone["sparse_index"],
        namespace=pinecone["namespace"],
        pinecone_cloud=pinecone["cloud"],
        pinecone_region=pinecone["region"],
        dense_dimension=int(pinecone["dense_dimension"]),
        dense_metric=pinecone["dense_metric"],
        sparse_model=pinecone["sparse_model"],
        dense_top_k=int(retrieval["dense_top_k"]),
        sparse_top_k=int(retrieval["sparse_top_k"]),
        final_top_k=int(retrieval["final_top_k"]),
        rrf_k=int(retrieval["rrf_k"]),
        api_host=api["host"],
        api_port=int(api["port"]),
        batch_size=int(config["batch_size"]),
        rag_cache_ttl_seconds=int(cache.get("rag_ttl_seconds", 3600)),
        crm_db_path=root / database.get("crm_path", "data/crm.db"),
        max_chunk_size=int(chunking["max_chunk_size"]),
        min_chunk_size=int(chunking["min_chunk_size"]),
        tag_topics=bool(chunking["tag_topics"]),
        chunk_skip_patterns=tuple(chunking["skip_patterns"]),
        table_extraction_enabled=bool(table_extraction["enabled"]),
        table_min_rows=int(table_extraction["min_rows"]),
        table_min_cols=int(table_extraction["min_cols"]),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


settings = load_settings()
