"""Tests for the section-aware ingestion pipeline."""
import pymupdf

from src.ingestion.chunker import chunk_page, detect_topics
from src.ingestion.page_parser import parse_page
from src.ingestion.table_classifier import classify_table


def _table(rows):
    """Build the table dict shape that page_parser hands to the classifier."""
    return {"rows": rows, "header": rows[0]}


# --- table_classifier ---------------------------------------------------
def test_classifier_skips_ombudsman_table():
    table = _table([
        ["Office of the Insurance Ombudsman", "Jurisdiction of Office"],
        ["Bimalokpal, Ahmedabad", "Gujarat, Dadra & Nagar Haveli"],
    ])

    assert classify_table(table)["destination"] == "skip"


def test_classifier_skips_non_payable_items_table():
    table = _table([
        ["Item", "Charge"],
        ["Baby food", "Not payable"],
        ["Hair removal cream", "Not payable"],
    ])

    assert classify_table(table)["destination"] == "skip"


def test_classifier_tags_cataract_table_as_sub_limit():
    table = _table([
        ["Treatment", "Sub-limit"],
        ["Cataract", "Rs 25,000 per eye"],
    ])

    assert classify_table(table)["table_type"] == "sub_limit"


def test_classifier_tags_waiting_period_table():
    table = _table([
        ["Ailment / Disease", "Waiting Period"],
        ["Benign ENT disorders", "24 months"],
    ])

    assert classify_table(table)["table_type"] == "waiting_period"


def test_classifier_tags_age_band_table_as_premium_rate():
    table = _table([
        ["SI/Age", "21-35 Yrs", "36-45 Yrs"],
        ["300000", "4500", "5200"],
    ])

    assert classify_table(table)["table_type"] == "premium_rate"


def test_classifier_skips_a_numbered_item_list_with_no_amounts():
    table = _table([
        ["S.No", "Item"],
        ["1", "Mortuary charges"],
        ["2", "Walking aids charges"],
        ["3", "Trolly cover"],
    ])

    assert classify_table(table)["destination"] == "skip"


def test_classifier_keeps_a_numbered_list_that_quotes_amounts():
    table = _table([
        ["S.No", "Treatment", "Sub-limit"],
        ["1", "Cataract", "Rs 25,000"],
        ["2", "Hernia", "Rs 30,000"],
    ])

    assert classify_table(table)["destination"] != "skip"


def test_classifier_tags_a_cancellation_grid():
    table = _table([
        ["Timing of cancellation", "Refund"],
        ["Up to 30 days", "75.00%"],
    ])

    assert classify_table(table)["table_type"] == "cancellation_refund"


def test_premium_rate_table_never_reaches_pinecone():
    table = _table([
        ["SI/Age", "21-35 Yrs"],
        ["300000", "4500"],
    ])

    assert classify_table(table)["destination"] == "sql_only"


# --- chunker ------------------------------------------------------------
def test_chunker_does_not_end_a_chunk_mid_sentence():
    text = (
        "4. EXCLUSIONS\n"
        + "The Company shall not be liable for any dental treatment. " * 12
        + "Cosmetic surgery is excluded from this policy."
    )

    chunks = chunk_page(text, "UIN1", "Star Health", "Arogya", 3)

    assert all(chunk["text"].rstrip().endswith(".") for chunk in chunks)


def test_chunker_tags_not_covered_text_as_exclusion():
    text = (
        "4.1 Dental treatment\n"
        "Dental treatment of any kind is not covered under this policy "
        "unless it arises from an accident requiring hospitalisation."
    )

    chunks = chunk_page(text, "UIN1", "Star Health", "Arogya", 4)

    assert chunks[0]["metadata"]["topic"] == "exclusion"


def test_chunker_tags_waiting_period_text():
    text = (
        "4.2 Specific waiting period\n"
        "A waiting period of 24 months shall apply to cataract surgery "
        "counted from the inception date of the first policy."
    )

    chunks = chunk_page(text, "UIN1", "Star Health", "Arogya", 5)

    assert chunks[0]["metadata"]["topic"] == "waiting_period"


def test_chunker_drops_toll_free_boilerplate():
    text = (
        "Toll free 1800 425 2255\n"
        "The insured person shall notify the company within 24 hours of "
        "an emergency hospitalisation to be eligible for cashless treatment."
    )

    chunks = chunk_page(text, "UIN1", "Star Health", "Arogya", 6)

    assert "1800" not in chunks[0]["text"]


def test_chunker_carries_uin_and_page_into_metadata():
    text = (
        "3. COVERAGE\n"
        "Hospitalisation expenses are payable up to the sum insured shown "
        "in the policy schedule for any admissible claim under this cover."
    )

    chunks = chunk_page(text, "SHAHLIP22027V032122", "Star Health", "Arogya", 9)

    assert chunks[0]["metadata"]["uin"] == "SHAHLIP22027V032122"


def test_detect_topic_falls_back_to_general():
    # detect_topics()[0] is what chunk_page() actually calls to fill the
    # single-string "topic" metadata field -- a wrapper for just the first
    # element used to sit between them and be tested here instead, which
    # meant this test could pass while the real call site had drifted.
    assert detect_topics("The schedule is attached herewith.")[0] == "general"


def test_detect_topic_does_not_tag_shall_not_apply_as_exclusion():
    # "shall not apply" is a scope statement, not an exclusion -- a bare
    # "shall not" keyword previously mistagged clauses like this one,
    # which made the eligibility engine misread genuine coverage clauses
    # as exclusions.
    text = "The time limit shall not apply in respect of Day Care Treatment."

    assert detect_topics(text)[0] == "general"


def test_detect_topic_tags_shall_not_be_liable_as_exclusion():
    text = "The Company shall not be liable for dental treatment of any kind."

    assert detect_topics(text)[0] == "exclusion"


# --- page_parser --------------------------------------------------------
def _page_with_text(body: str):
    """Build a one-page in-memory PDF so the tests need no temp files."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), body)
    return document, page


def test_page_parser_returns_no_tables_for_a_plain_text_page():
    document, page = _page_with_text("This page has prose and no tables.")

    _, tables = parse_page(page)
    document.close()

    assert tables == []


def test_page_parser_returns_the_page_text():
    document, page = _page_with_text("Cataract surgery is covered.")

    text, _ = parse_page(page)
    document.close()

    assert "Cataract surgery" in text


# --- table type -> retrieval topics ---------------------------------------
# WHY these exist: a table's type names the shape of the table in the PDF
# ("modern_treatment"), while the RAG tools filter on the topic vocabulary the
# chunker uses for prose ("coverage"). The pipeline used to write the table
# type straight into `topics`, so table rows were indexed under labels no tool
# ever asks for -- present in Pinecone, correct, citable, and unreachable.
def test_every_table_type_maps_onto_a_topic_a_tool_searches_for():
    """The two vocabularies have to meet, or whole tables go missing."""
    from src.ingestion.table_classifier import TABLE_TYPE_TOPICS
    from src.tools.rag_tools import COVERAGE_TOPICS

    searchable = set(COVERAGE_TOPICS) | {
        "exclusion", "waiting_period", "definition", "claim_procedure", "general",
    }

    for table_type, topics in TABLE_TYPE_TOPICS.items():
        assert topics, f"{table_type} maps to no topic at all"
        unreachable = set(topics) - searchable
        assert not unreachable, (
            f"{table_type} is indexed under {unreachable}, which no tool filters on"
        )


def test_modern_treatment_rows_are_findable_as_coverage():
    """The regression case: a robotic-surgery row must answer a coverage query."""
    from src.ingestion.table_classifier import retrieval_topics
    from src.tools.rag_tools import COVERAGE_TOPICS

    assert set(retrieval_topics("modern_treatment")) & set(COVERAGE_TOPICS)


def test_copayment_rows_are_findable_by_the_coverage_tool():
    """The engine resolves co-pay through check_coverage, so it has to reach them."""
    from src.ingestion.table_classifier import retrieval_topics
    from src.tools.rag_tools import COVERAGE_TOPICS

    assert set(retrieval_topics("copayment")) & set(COVERAGE_TOPICS)


def test_an_unmapped_table_type_falls_back_to_general_not_to_itself():
    """Better unfiltered than filed under a label nothing searches for."""
    from src.ingestion.table_classifier import retrieval_topics

    assert retrieval_topics("some_new_table_shape") == ["general"]
