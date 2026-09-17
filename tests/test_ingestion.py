"""Tests for the generic PDF ingestion pipeline."""

import pymupdf

from src.ingestion.chunker import chunk_page, detect_topics
from src.ingestion.page_parser import parse_page
from src.ingestion.pipeline import extract_product, extract_uin
from src.ingestion.table_classifier import classify_table, retrieval_topics
from src.ingestion.table_converter import table_to_sentences


def _table(rows):
    """Build the table shape returned by page_parser."""
    return {"rows": rows, "header": rows[0]}


def test_extract_uin_from_filename():
    assert extract_uin("SHAHLIP22027V032122_HEALTH.pdf", "") == "SHAHLIP22027V032122"


def test_missing_uin_is_allowed():
    assert extract_uin("new_policy.pdf", "No UIN printed here") == ""


def test_product_uses_pdf_title_when_available():
    assert extract_product("fallback_name.pdf", {"title": "Health Protect Plus"}) == "Health Protect Plus"


def test_product_falls_back_to_filename():
    assert extract_product("health_policy_v2.pdf", {}) == "health policy v2"


def test_classifier_tags_waiting_period_table():
    table = _table([
        ["Condition", "Waiting Period"],
        ["Cataract", "24 months"],
    ])

    result = classify_table(table)
    assert result["table_type"] == "waiting_period"
    assert result["destination"] == "rag_and_sql"


def test_classifier_tags_sub_limit_table():
    table = _table([
        ["Treatment", "Sub-limit"],
        ["Cataract", "Rs 25,000 per eye"],
    ])

    result = classify_table(table)
    assert result["table_type"] == "sub_limit"
    assert result["destination"] == "rag_and_sql"


def test_classifier_tags_copayment_table():
    table = _table([
        ["Condition", "Co-payment"],
        ["Age above 60", "20%"],
    ])

    assert classify_table(table)["table_type"] == "copayment"


def test_classifier_tags_premium_grid():
    table = _table([
        ["SI/Age", "21-35", "36-45"],
        ["300000", "4500", "5200"],
    ])

    result = classify_table(table)
    assert result["table_type"] == "premium_rate"
    assert result["destination"] == "rag_and_sql"


def test_unknown_table_is_preserved_for_rag():
    table = _table([
        ["Benefit", "Description"],
        ["Home nursing", "Available after discharge"],
    ])

    assert classify_table(table) == {
        "table_type": "generic_table",
        "destination": "rag_only",
    }


def test_non_payable_table_is_not_silently_dropped():
    table = _table([
        ["Item", "Status"],
        ["Walking aids", "Not payable"],
        ["Registration charges", "Not payable"],
    ])

    assert classify_table(table)["destination"] == "rag_only"


def test_unmapped_table_type_uses_general_topic():
    assert retrieval_topics("brand_new_table_shape") == ["general"]


def test_generic_table_sentence_keeps_header_meaning():
    table = _table([
        ["Benefit", "Limit"],
        ["Ambulance", "Rs 5,000"],
    ])

    sentences = table_to_sentences(
        table,
        "generic_table",
        insurer="",
        uin="",
        page=3,
    )

    assert len(sentences) == 1
    assert "Benefit: Ambulance" in sentences[0]
    assert "Limit: Rs 5,000" in sentences[0]
    assert "page 3" in sentences[0]


def test_chunker_does_not_end_a_chunk_mid_sentence():
    text = (
        "4. EXCLUSIONS\n"
        + "The Company shall not be liable for any dental treatment. " * 12
        + "Cosmetic surgery is excluded from this policy."
    )

    chunks = chunk_page(text, "UIN1", "", "Arogya", 3)

    assert all(chunk["text"].rstrip().endswith(".") for chunk in chunks)


def test_chunker_tags_not_covered_text_as_exclusion():
    text = (
        "4.1 Dental treatment\n"
        "Dental treatment of any kind is not covered under this policy "
        "unless it arises from an accident requiring hospitalisation."
    )

    chunks = chunk_page(text, "UIN1", "", "Arogya", 4)

    assert chunks[0]["metadata"]["topic"] == "exclusion"


def test_chunker_tags_waiting_period_text():
    text = (
        "4.2 Specific waiting period\n"
        "A waiting period of 24 months shall apply to cataract surgery."
    )

    chunks = chunk_page(text, "UIN1", "", "Arogya", 5)

    assert chunks[0]["metadata"]["topic"] == "waiting_period"


def test_chunker_allows_empty_optional_metadata():
    text = (
        "3. COVERAGE\n"
        "Hospitalisation expenses are payable up to the sum insured."
    )

    chunks = chunk_page(text, "", "", "New Policy", 9)

    assert chunks[0]["metadata"]["uin"] == ""
    assert chunks[0]["metadata"]["insurer"] == ""
    assert chunks[0]["metadata"]["product"] == "New Policy"


def test_detect_topic_falls_back_to_general():
    assert detect_topics("The schedule is attached herewith.")[0] == "general"


def test_detect_topic_does_not_tag_shall_not_apply_as_exclusion():
    text = "The time limit shall not apply in respect of Day Care Treatment."
    assert detect_topics(text)[0] == "general"


def test_detect_topic_tags_shall_not_be_liable_as_exclusion():
    text = "The Company shall not be liable for dental treatment of any kind."
    assert detect_topics(text)[0] == "exclusion"


def _page_with_text(body: str):
    """Build a one-page in-memory PDF for parser tests."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), body)
    return document, page


def test_page_parser_returns_no_tables_for_plain_text():
    document, page = _page_with_text("This page has prose and no tables.")
    _, tables = parse_page(page)
    document.close()

    assert tables == []


def test_page_parser_returns_page_text():
    document, page = _page_with_text("Cataract surgery is covered.")
    text, _ = parse_page(page)
    document.close()

    assert "Cataract surgery" in text
