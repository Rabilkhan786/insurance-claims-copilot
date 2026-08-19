"""PDF ingestion through the existing Unstructured API and title chunker."""
from pathlib import Path

from config import settings


def load_pdf_documents(pdf_directory: Path | None = None) -> list:
    """Partition PDFs via Unstructured, chunk by title, drop tiny chunks."""
    if not settings.unstructured_api_key:
        raise RuntimeError("UNSTRUCTURED_API_KEY is required for ingestion")

    from unstructured.chunking.title import chunk_by_title
    from unstructured.staging.base import elements_from_dicts
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import operations, shared

    from langchain_core.documents import Document

    client = UnstructuredClient(api_key_auth=settings.unstructured_api_key)
    documents: list[Document] = []
    source_directory = pdf_directory or settings.data_dir

    for pdf_path in sorted(source_directory.glob("*.pdf")):
        with pdf_path.open("rb") as source:
            files = shared.Files(
                content=source.read(),
                file_name=pdf_path.name,
            )
            parameters = shared.PartitionParameters(
                files=files,
                strategy=shared.Strategy.HI_RES,
                coordinates=True,
                infer_table_structure=True,
            )
            request = operations.PartitionRequest(
                partition_parameters=parameters,
            )

        response = client.general.partition(request=request)
        elements = elements_from_dicts(response.elements)
        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=settings.chunk_combine_under,
            new_after_n_chars=settings.chunk_new_after,
            max_characters=settings.chunk_max_characters,
            multipage_sections=False,
        )

        for chunk in chunks:
            text = chunk.text.strip()
            if len(text) < settings.chunk_min_characters:
                continue

            metadata = (
                chunk.metadata.to_dict()
                if hasattr(chunk.metadata, "to_dict")
                else {}
            )
            metadata["source"] = pdf_path.name
            metadata["category"] = chunk.category
            documents.append(
                Document(page_content=text, metadata=metadata)
            )

    return documents
