from __future__ import annotations

import fitz
import pytest

import parser as parser_mod
from parser import AcademicPaperParser, ExtractedTable, ParsedPaper, estimate_tokens


def test_estimate_tokens_empty_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("Attention is all you need.") > 0

    monkeypatch.setattr(parser_mod, "_ENCODER", None)
    assert estimate_tokens("a" * 38) == 10
    assert estimate_tokens("x") == 1


def test_normalize_unwraps_midword_hyphens_and_preserves_compounds() -> None:
    p = AcademicPaperParser()
    sample = (
        "We present an architec-\n"
        "ture for computa-\n"
        "tional optimiza-\n"
        "tion in re-\n"
        "search and in-\n"
        "formation retrieval across multi-\n"
        "ple high-\n"
        "er benchmarks.\n\n"
        "It uses self-\n"
        "attention, multi-\n"
        "head projections, fine-\n"
        "tuning, zero-\n"
        "shot transfer, low-\n"
        "rank adapters, cross-\n"
        "entropy loss, pre-\n"
        "training, in-\n"
        "context learning, token-\n"
        "level routing, memory-\n"
        "efficient kernels, KV-\n"
        "cache compression, end-\n"
        "to-end training, and state-\n"
        "of-the-art state-of-\n"
        "the-art accuracy.\n\n"
        "Our custom zebra-flux module also handles zebra-\n"
        "flux scheduling."
    )

    out = p._normalize_equations_and_whitespace(sample)

    # Genuine mid-word line-break hyphens unwrapped
    for expected in (
        "architecture",
        "computational",
        "optimization",
        "research",
        "information",
        "multiple",
        "higher",
    ):
        assert expected in out

    # Genuine hyphenated compounds preserved
    for expected in (
        "self-attention",
        "multi-head",
        "fine-tuning",
        "zero-shot",
        "low-rank",
        "cross-entropy",
        "pre-training",
        "in-context",
        "token-level",
        "memory-efficient",
        "KV-cache",
        "end-to-end",
        "state-of-the-art",
        "zebra-flux",
    ):
        assert expected in out


def test_normalize_preserves_code_fences_tables_and_equations() -> None:
    p = AcademicPaperParser()
    sample = (
        "```python\n"
        "architec-\n"
        "ture = 1\n"
        "```\n\n"
        "| Metric | Value |\n"
        "| --- | --- |\n"
        "| architec- | 99.1 |\n"
        "| ture | 98.2 |\n\n"
        "E = mc^2\n"
        "(1)\n"
        "Next paragraph."
    )

    out = p._normalize_equations_and_whitespace(sample)
    assert "architec-\nture = 1" in out
    assert "| architec- | 99.1 |" in out
    assert "`[Eq. (1)]`" in out


def test_classify_heading_recognizes_numbered_and_markdown_headings() -> None:
    p = AcademicPaperParser()
    assert not hasattr(AcademicPaperParser, "HEADING_LINE_RE")
    assert p._classify_heading("# Abstract") == "Abstract"
    assert p._classify_heading("## 1. Introduction") == "Introduction"
    assert p._classify_heading("1. Introduction") == "Introduction"
    assert p._classify_heading("3 Methodology") == "Methodology"
    assert p._classify_heading("IV. Experiments and Results") == "Experiments & Benchmarks"
    assert p._classify_heading("**Limitations and Future Work**") == "Limitations"
    assert p._classify_heading("References") == "References"
    assert p._classify_heading("A. Appendix") == "Appendix"
    assert p._classify_heading("A.1 Proofs") == "Appendix"
    # Table rows and normal sentences are not headings
    assert p._classify_heading("| 1. Introduction | 94.2 |") is None
    assert p._classify_heading("This introduction discusses attention in detail.") is None
    assert p._classify_heading("A Method for Fast Attention") is None


def test_segment_does_not_hardcode_line_140_abstract_cutoff() -> None:
    p = AcademicPaperParser()
    # 160 lines of authors/affiliations before explicit Abstract and Introduction
    preamble = "\n".join(f"Author {i}, University {i}" for i in range(160))
    abstract_lines = "\n".join(f"word{i} token{i}" for i in range(150))
    raw_md = (
        f"{preamble}\n\n"
        "## Abstract\n"
        f"{abstract_lines}\n\n"
        "## 1. Introduction\n"
        "Actual introduction text starts here.\n\n"
        "## 2. Methodology\n"
        "Proposed method details.\n\n"
        "## References\n"
        "[1] Vaswani et al., 2017.\n"
    )

    sections, refs, cleaned = p._segment_and_strip_references(raw_md, strip_references=True)

    assert "Abstract" in sections
    assert "word149 token149" in sections["Abstract"]
    assert "word149" not in sections["Introduction"]
    assert "Actual introduction text starts here." in sections["Introduction"]
    assert "Methodology" in sections
    assert "[1] Vaswani et al., 2017." in refs
    assert "Vaswani et al." not in cleaned


def test_segment_auto_transitions_long_unheaded_abstract_at_paragraph_boundary() -> None:
    p = AcademicPaperParser()
    para1 = " ".join(f"absword{i}" for i in range(420))
    para2 = "Unheaded introduction paragraph discussing motivation and contributions."
    raw_md = (
        "## Abstract\n"
        f"{para1}\n\n"
        f"{para2}\n\n"
        "## 2. Methodology\n"
        "Algorithmic formulation goes here.\n"
    )

    sections, _, _ = p._segment_and_strip_references(raw_md, strip_references=True)

    assert "absword419" in sections["Abstract"]
    assert "Unheaded introduction paragraph" in sections["Introduction"]
    assert "Unheaded introduction paragraph" not in sections["Abstract"]
    assert "Methodology" in sections


def test_parse_pdf_end_to_end_with_in_memory_pdf() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "ScalpelFormer: Sparse Routing for Long Context", fontsize=18)
    body = (
        "Abstract\n"
        "We propose ScalpelFormer using self-attention and KV-cache compression.\n\n"
        "1. Introduction\n"
        "Transformer inference is bounded by memory bandwidth.\n\n"
        "2. Methodology\n"
        "We introduce low-rank projections and sparse routing.\n\n"
        "3. Experiments and Results\n"
        "ScalpelFormer improves MMLU from 72.1 to 76.4.\n\n"
        "References\n"
        "[1] Doe et al., Attention Benchmarks, 2024.\n"
        "[2] Smith et al., Sparse Transformers, 2025.\n"
    )
    page.insert_textbox(fitz.Rect(72, 110, 520, 760), body, fontsize=11)
    doc.set_metadata({"title": "ScalpelFormer: Sparse Routing for Long Context"})
    pdf_bytes = doc.tobytes()
    doc.close()

    parsed = AcademicPaperParser().parse_pdf(pdf_bytes, strip_references=True)

    assert isinstance(parsed, ParsedPaper)
    assert parsed.title == "ScalpelFormer: Sparse Routing for Long Context"
    assert parsed.page_count == 1
    assert "Abstract" in parsed.sections
    assert "Introduction" in parsed.sections
    assert "Methodology" in parsed.sections
    assert "Experiments & Benchmarks" in parsed.sections
    assert "Doe et al." in parsed.references_text
    assert "Doe et al." not in parsed.cleaned_markdown
    assert parsed.total_tokens_raw >= parsed.total_tokens_cleaned > 0


def test_tables_markdown_and_validation() -> None:
    p = AcademicPaperParser()
    valid_tbl = (
        "| Model | MMLU | Latency (ms) |\n"
        "| --- | --- | --- |\n"
        "| Baseline | 71.2 | 42.0 |\n"
        "| Scalpel | 75.8 | 21.5 |"
    )
    trivial_tbl = "| a |\n| --- |"
    assert p._is_valid_benchmark_table(valid_tbl) is True
    assert p._is_valid_benchmark_table(trivial_tbl) is False

    paper = ParsedPaper(
        title="Demo",
        authors_or_header="",
        page_count=1,
        raw_markdown="",
        cleaned_markdown="",
        tables=[
            ExtractedTable(
                table_id="Table-1",
                page_number=2,
                caption="Table 1: Main Benchmark",
                markdown=valid_tbl,
                row_count=3,
                col_count=3,
            )
        ],
    )
    digest = paper.get_tables_markdown()
    assert "Table-1 | Page 2" in digest
    assert "Table 1: Main Benchmark" in digest
    assert "75.8" in digest
