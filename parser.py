"""
parser.py
Layout-aware academic PDF parser built on PyMuPDF (fitz) and pymupdf4llm.
Handles two-column reading order, explicit benchmark table extraction,
canonical academic section segmentation, and bibliography stripping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pymupdf4llm

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None


CANONICAL_SECTIONS: Tuple[str, ...] = (
    "Abstract",
    "Introduction",
    "Related Work",
    "Methodology",
    "Experiments & Benchmarks",
    "Limitations",
    "Conclusion",
    "Appendix",
)


@dataclass
class ExtractedTable:
    table_id: str
    page_number: int
    caption: str
    markdown: str
    row_count: int
    col_count: int


@dataclass
class ParsedPaper:
    title: str
    authors_or_header: str
    page_count: int
    raw_markdown: str
    cleaned_markdown: str
    sections: Dict[str, str] = field(default_factory=dict)
    references_text: str = ""
    tables: List[ExtractedTable] = field(default_factory=list)
    total_tokens_raw: int = 0
    total_tokens_cleaned: int = 0
    reference_tokens_saved: int = 0

    def get_tables_markdown(self) -> str:
        """Formats all explicitly extracted tables into a single Markdown digest."""
        if not self.tables:
            return "No explicit grid tables detected by PyMuPDF table finder."
        blocks: List[str] = []
        for tbl in self.tables:
            caption_str = f" — {tbl.caption}" if tbl.caption else ""
            blocks.append(
                f"### [{tbl.table_id} | Page {tbl.page_number}{caption_str}]\n"
                f"*(Dimensions: {tbl.row_count} rows x {tbl.col_count} cols)*\n\n"
                f"{tbl.markdown}"
            )
        return "\n\n---\n\n".join(blocks)


def estimate_tokens(text: str) -> int:
    """Estimates LLM token count using cl100k_base or a conservative heuristic."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Fallback: ~3.8 characters per token in dense technical/LaTeX text
    return max(1, int(len(text) / 3.8))


class AcademicPaperParser:
    """
    Extracts structured Markdown, canonical sections, and benchmark tables
    from multi-column research PDFs while stripping reference lists.
    """

    # Maps normalized academic heading patterns to canonical buckets
    SECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
        (
            "Abstract",
            re.compile(r"^\s*abstract\b", re.IGNORECASE),
        ),
        (
            "Introduction",
            re.compile(
                r"^\s*(?:[0-9IVX]+[\.\s]+)?(?:introduction|background\s+and\s+motivation|overview)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "Related Work",
            re.compile(
                r"^\s*(?:[0-9IVX]+[\.\s]+)?(?:related\s+works?|prior\s+works?|literature\s+review|preliminaries|background)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "Methodology",
            re.compile(
                r"^\s*(?:[0-9IVX]+[\.\s]+)?(?:method(?:ology|s)?|proposed\s+(?:method|approach|framework|architecture|system)|approach|model(?:\s+architecture)?|formulation|technical\s+approach|system\s+design|algorithm)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "Experiments & Benchmarks",
            re.compile(
                r"^\s*(?:[0-9IVX]+[\.\s]+)?(?:experiments?(?:\s+and\s+results)?|empirical\s+evaluation|evaluation|results?(?:\s+and\s+discussion|\s+and\s+analysis)?|benchmarks?|ablation\s+stud(?:y|ies)|performance\s+analysis)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "Limitations",
            re.compile(
                r"^\s*(?:[0-9IVX]+[\.\s]+)?(?:limitations?(?:\s+and\s+future\s+work)?|broader\s+impact|discussion(?:\s+and\s+limitations)?|threats\s+to\s+validity|failure\s+modes)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "Conclusion",
            re.compile(
                r"^\s*(?:[0-9IVX]+[\.\s]+)?(?:conclusions?(?:\s+and\s+future\s+work)?|concluding\s+remarks|summary(?:\s+and\s+conclusion)?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "References",
            re.compile(
                r"^\s*(?:[0-9IVX]+[\.\s]+)?(?:references|bibliography|works\s+cited)\s*$",
                re.IGNORECASE,
            ),
        ),
        (
            "Appendix",
            re.compile(
                r"^\s*(?:[A-Z0-9]+[\.\s]+)?(?:appendix|appendices|supplementary\s+material|additional\s+(?:results|experiments|details)|proofs?)\b",
                re.IGNORECASE,
            ),
        ),
    ]

    HEADING_LINE_RE = re.compile(
        r"^(?:#{1,4}\s+|\*\*(?=[0-9IVX]+\b|Abstract\b|References\b|Appendix\b))?(?P<title>[^\n]{2,110}?)(?:\*\*)?\s*$"
    )

     numbered_section_re = re.compile(
        r"^(?:#{1,4}\s*)?(?:[0-9]{1,2}|[IVX]{1,4})(?:\.[0-9]{1,2})*\s+[A-Z][A-Za-z0-9\s\-:,/&]{2,80}$"
    )

    TABLE_CAPTION_RE = re.compile(
        r"(Table\s+\d+[\s\.:\-—]+[^\n]{5,220})",
        re.IGNORECASE,
    )

    def parse_pdf(self, pdf_bytes: bytes, strip_references: bool = True) -> ParsedPaper:
        """
        Full pipeline:
        1. Open PDF in fitz (PyMuPDF).
        2. Run pymupdf4llm for two-column reading-order Markdown conversion.
        3. Run explicit geometric + text table extraction across all pages.
        4. Segment paper into canonical sections and isolate References.
        5. Compute token telemetry.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_count = len(doc)
            title, authors_header = self._extract_title_and_header(doc)

            # Convert multi-column layout to reading-order Markdown
            raw_md = pymupdf4llm.to_markdown(
                doc,
                write_images=False,
                table_strategy="lines_strict",
            )
            raw_md = self._normalize_equations_and_whitespace(raw_md)

            # Explicitly extract benchmark/ablation tables
            tables = self._extract_all_tables(doc, raw_md)

            # Segment into logical sections and isolate references
            sections, references_text, cleaned_md = self._segment_and_strip_references(
                raw_md, strip_references=strip_references
            )

            total_tokens_raw = estimate_tokens(raw_md)
            total_tokens_cleaned = estimate_tokens(cleaned_md)
            ref_tokens_saved = max(0, total_tokens_raw - total_tokens_cleaned)

            return ParsedPaper(
                title=title,
                authors_or_header=authors_header,
                page_count=page_count,
                raw_markdown=raw_md,
                cleaned_markdown=cleaned_md,
                sections=sections,
                references_text=references_text,
                tables=tables,
                total_tokens_raw=total_tokens_raw,
                total_tokens_cleaned=total_tokens_cleaned,
                reference_tokens_saved=ref_tokens_saved,
            )
        finally:
            doc.close()

    def _extract_title_and_header(self, doc: fitz.Document) -> Tuple[str, str]:
        """Extracts paper title from PDF metadata or largest font span on page 1."""
        meta_title = (doc.metadata or {}).get("title", "").strip()
        if (
            meta_title
            and len(meta_title) > 8
            and not meta_title.lower().startswith("untitled")
            and not meta_title.lower().endswith(".pdf")
            and "arxiv" not in meta_title.lower()
        ):
            first_page_text = doc[0].get_text("text") if len(doc) > 0 else ""
            header_preview = " ".join(first_page_text.splitlines()[:6]).strip()
            return meta_title, header_preview[:300]

        if len(doc) == 0:
            return "Untitled Research Paper", ""

        first_page = doc[0]
        blocks = first_page.get_text("dict").get("blocks", [])
        max_size = 0.0
        title_candidates: List[str] = []

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                # Ignore vertical arXiv stamps on the far left margin
                bbox = line.get("bbox", [0, 0, 0, 0])
                if bbox[0] < 38 and (bbox[3] - bbox[1]) > 100:
                    continue
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = float(span.get("size", 0.0))
                    if len(text) < 3 or text.lower().startswith("arxiv:"):
                        continue
                    if size > max_size + 0.5:
                        max_size = size
                        title_candidates = [text]
                    elif abs(size - max_size) <= 0.5:
                        title_candidates.append(text)

        title = " ".join(title_candidates).strip()
        if not title:
            lines = [
                ln.strip()
                for ln in first_page.get_text("text").splitlines()
                if len(ln.strip()) > 5 and "arxiv:" not in ln.lower()
            ]
            title = lines[0] if lines else "Untitled Research Paper"

        raw_top_lines = [
            ln.strip()
            for ln in first_page.get_text("text").splitlines()[:12]
            if ln.strip()
        ]
        header_preview = " | ".join(raw_top_lines[1:5])[:300]
        return title[:220], header_preview

    def _normalize_equations_and_whitespace(self, md_text: str) -> str:
        """Cleans hyphenated line breaks and normalizes LaTeX display blocks."""
        # Collapse 3+ blank lines into 2
        cleaned = re.sub(r"\n{3,}", "\n\n", md_text)
        # Rejoin words hyphenated across column wraps outside code/tables
        cleaned = re.sub(r"([a-z]{2,})-\n([a-z]{2,})", r"\1\2", cleaned)
        # Normalize numbered equation markers like "(1)" at end of lines next to math
        cleaned = re.sub(
            r"\n\s*\((\d{1,2}[a-z]?)\)\s*\n",
            r"  `[Eq. (\1)]`\n\n",
            cleaned,
        )
        return cleaned.strip()

    def _extract_all_tables(
        self, doc: fitz.Document, full_markdown: str
    ) -> List[ExtractedTable]:
        """
        Combines PyMuPDF geometric table detection (`page.find_tables()`)
        with Markdown pipe-table extraction so numeric benchmark tables are
        preserved even on borderless Booktabs LaTeX layouts.
        """
        extracted: List[ExtractedTable] = []
        seen_signatures = set()
        table_counter = 1

        # Pass 1: PyMuPDF native geometric table finder per page
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            page_text = page.get_text("text")
            captions = self.TABLE_CAPTION_RE.findall(page_text)

            try:
                finder = page.find_tables()
                page_tables = finder.tables if finder else []
            except Exception:
                page_tables = []

            for idx, tbl in enumerate(page_tables):
                try:
                    md_table = tbl.to_markdown().strip()
                except Exception:
                    continue

                if not self._is_valid_benchmark_table(md_table):
                    continue

                sig = re.sub(r"\s+", "", md_table)[:180]
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)

                caption = captions[idx].strip() if idx < len(captions) else (
                    captions[0].strip() if captions else f"Detected Table on Page {page_num}"
                )
                rows = [r for r in md_table.splitlines() if r.strip().startswith("|")]
                col_count = max((r.count("|") - 1 for r in rows), default=0)

                extracted.append(
                    ExtractedTable(
                        table_id=f"Table-{table_counter}",
                        page_number=page_num,
                        caption=caption,
                        markdown=md_table,
                        row_count=max(1, len(rows) - 1),
                        col_count=max(1, col_count),
                    )
                )
                table_counter += 1

        # Pass 2: Markdown pipe-tables produced by pymupdf4llm not caught in Pass 1
        md_table_pattern = re.compile(
            r"(?:(?P<caption>Table\s+\d+[^\n]*)\n\n?)?"
            r"(?P<table>(?:\|[^\n]+\|\n)+(?:\|[\s:\-\|]+\|\n)(?:\|[^\n]+\|\n?)+)",
            re.IGNORECASE,
        )
        for match in md_table_pattern.finditer(full_markdown):
            md_table = match.group("table").strip()
            if not self._is_valid_benchmark_table(md_table):
                continue
            sig = re.sub(r"\s+", "", md_table)[:180]
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            caption = (match.group("caption") or "").strip() or f"Extracted Markdown Table {table_counter}"
            rows = [r for r in md_table.splitlines() if r.strip().startswith("|")]
            col_count = max((r.count("|") - 1 for r in rows), default=0)

            extracted.append(
                ExtractedTable(
                    table_id=f"Table-{table_counter}",
                    page_number=0,
                    caption=caption,
                    markdown=md_table,
                    row_count=max(1, len(rows) - 1),
                    col_count=max(1, col_count),
                )
            )
            table_counter += 1

        return extracted

    @staticmethod
    def _is_valid_benchmark_table(md_table: str) -> bool:
        """Filters out trivial 1x1 layout boxes or non-data artifacts."""
        lines = [ln.strip() for ln in md_table.splitlines() if ln.strip()]
        if len(lines) < 3:
            return False
        # Require at least some alphanumeric content and digits typical of metrics
        alnum_chars = sum(ch.isalnum() for ch in md_table)
        digit_chars = sum(ch.isdigit() for ch in md_table)
        return alnum_chars >= 25 and digit_chars >= 2

    def _classify_heading(self, line: str) -> Optional[str]:
        """Determines if a Markdown line is a major section heading and maps it."""
        stripped = line.strip()
        if not stripped or len(stripped) > 115 or stripped.startswith("|"):
            return None

        is_markdown_header = stripped.startswith("#")
        is_bold_line = (
            stripped.startswith("**")
            and stripped.endswith("**")
            and stripped.count("**") == 2
        )
        is_numbered_line = bool(self.numbered_section_re.match(stripped))

        # Clean markdown decorators
        clean_title = re.sub(r"^[#*_\s]+|[#*_\s]+$", "", stripped).strip()
        if not clean_title or len(clean_title) < 3:
            return None

        # Only inspect lines that visually act as headings or explicit Abstract/References
        if not (
            is_markdown_header
            or is_bold_line
            or is_numbered_line
            or clean_title.lower() in {"abstract", "references", "bibliography", "appendix"}
        ):
            return None

        for canonical_name, pattern in self.SECTION_PATTERNS:
            if pattern.search(clean_title):
                return canonical_name

        return None

    def _segment_and_strip_references(
        self, raw_md: str, strip_references: bool = True
    ) -> Tuple[Dict[str, str], str, str]:
        """
        Splits the paper into canonical sections, isolates References,
        and reconstructs a clean Markdown document without bibliography bloat.
        """
        buckets: Dict[str, List[str]] = {sec: [] for sec in CANONICAL_SECTIONS}
        references_lines: List[str] = []

        lines = raw_md.splitlines()
        current_section = "Abstract"
        seen_intro = False

        # Heuristic: look for explicit "Abstract" in the first 120 lines
        for idx, line in enumerate(lines):
            detected = self._classify_heading(line)
            if detected is not None:
                current_section = detected
                if detected == "Introduction":
                    seen_intro = True
            else:
                # If we are still in preamble/Abstract and hit > 350 words before Intro,
                # keep in Abstract until a heading transitions us.
                if current_section == "Abstract" and not seen_intro and idx > 140:
                    current_section = "Introduction"
                    seen_intro = True

            if current_section == "References":
                references_lines.append(line)
            else:
                buckets.setdefault(current_section, []).append(line)

        sections_dict: Dict[str, str] = {}
        for sec_name, sec_lines in buckets.items():
            content = "\n".join(sec_lines).strip()
            if content:
                sections_dict[sec_name] = content

        # Fallback if a paper had unconventional headings and everything landed in 1 bucket
        if len(sections_dict) <= 1 and len(raw_md) > 2000:
            sections_dict = self._fallback_positional_segmentation(raw_md)

        references_text = "\n".join(references_lines).strip()

        if strip_references:
            cleaned_parts = [
                f"## [{sec}]\n\n{text}"
                for sec, text in sections_dict.items()
                if text.strip()
            ]
            cleaned_md = "\n\n---\n\n".join(cleaned_parts)
        else:
            cleaned_md = raw_md

        return sections_dict, references_text, cleaned_md

    def _fallback_positional_segmentation(self, raw_md: str) -> Dict[str, str]:
        """
        Positional fallback when a PDF lacks standard section headings
        (e.g., workshop papers or scanned OCR text).
        """
        # Strip trailing references if present via regex search from the bottom
        ref_split = re.split(
            r"\n(?:#{1,4}\s*|\*\*)?(?:References|Bibliography)(?:\*\*)?\s*\n",
            raw_md,
            flags=re.IGNORECASE,
        )
        body = ref_split[0]
        n = len(body)
        return {
            "Abstract": body[: int(n * 0.08)].strip(),
            "Introduction": body[int(n * 0.08) : int(n * 0.25)].strip(),
            "Methodology": body[int(n * 0.25) : int(n * 0.60)].strip(),
            "Experiments & Benchmarks": body[int(n * 0.60) : int(n * 0.88)].strip(),
            "Conclusion": body[int(n * 0.88) :].strip(),
        }
