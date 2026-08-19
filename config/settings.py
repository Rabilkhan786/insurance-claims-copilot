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
    "memory",
    "voice",
    "api",
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
    rrf_top_k: int
    final_top_k: int
    rrf_k: int
    chunk_combine_under: int
    chunk_new_after: int
    chunk_max_characters: int
    chunk_min_characters: int
    memory_max_messages: int
    voice_model: str
    voice_language: str
    voice_sample_rate: int
    api_host: str
    api_port: int
    batch_size: int
    pinecone_api_key: str | None
    groq_api_key: str | None
    unstructured_api_key: str | None
    hf_token: str | None


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


def load_settings() -> Settings:
    """Load env vars and config.yaml into a validated Settings object."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    raw = _load_yaml_config(root)

    project = raw["project"]
    models = raw["models"]
    pinecone = raw["pinecone"]
    retrieval = raw["retrieval"]
    chunking = raw["chunking"]
    memory = raw["memory"]
    voice = raw["voice"]
    api = raw["api"]

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
        rrf_top_k=int(retrieval["rrf_top_k"]),
        final_top_k=int(retrieval["final_top_k"]),
        rrf_k=int(retrieval["rrf_k"]),
        chunk_combine_under=int(chunking["combine_text_under_n_chars"]),
        chunk_new_after=int(chunking["new_after_n_chars"]),
        chunk_max_characters=int(chunking["max_characters"]),
        chunk_min_characters=int(chunking["min_characters"]),
        memory_max_messages=int(memory["max_messages"]),
        voice_model=voice["model"],
        voice_language=voice["language"],
        voice_sample_rate=int(voice["sample_rate"]),
        api_host=api["host"],
        api_port=int(api["port"]),
        batch_size=int(raw["batch_size"]),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        unstructured_api_key=os.getenv("UNSTRUCTURED_API_KEY"),
        hf_token=os.getenv("HF_TOKEN"),
    )


settings = load_settings()
