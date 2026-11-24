"""
analyzer.py
Orchestrates the 4 deep technical teardown modules and paper Q&A.
Implements both Full-Context execution and VRAM-safe Section-Routed Map-Reduce.
"""

from __future__ import annotations

import json
import re
from collections.abc import Generator
from datetime import datetime, timezone
from enum import Enum

from llm_engine import ExecutionStrategy, HybridLLMEngine
from parser import ParsedPaper, estimate_tokens
from prompts import (
    BASE_SYSTEM_PROMPT,
    PROMPT_BENCHMARKS_AND_ABLATIONS,
    PROMPT_BOTTLENECKS_AND_CRITIQUE,
    PROMPT_CHAT_SYSTEM,
    PROMPT_EXECUTIVE_SUMMARY,
    PROMPT_TOPICS_AND_MATH,
    build_module_user_prompt,
)

_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_CHAT_STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "can",
        "could",
        "did",
        "does",
        "doing",
        "during",
        "each",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "here",
        "how",
        "into",
        "just",
        "many",
        "more",
        "most",
        "much",
        "not",
        "once",
        "only",
        "other",
        "our",
        "over",
        "same",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "until",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


class AnalysisModule(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    TOPICS_AND_MATH = "topics_and_math"
    BENCHMARKS_AND_ABLATIONS = "benchmarks_and_ablations"
    BOTTLENECKS_AND_CRITIQUE = "bottlenecks_and_critique"


MODULE_TITLES: dict[AnalysisModule, str] = {
    AnalysisModule.EXECUTIVE_SUMMARY: "1. Executive Summary & Core Delta",
    AnalysisModule.TOPICS_AND_MATH: "2. Important Topics, Taxonomy & Math Formulation",
    AnalysisModule.BENCHMARKS_AND_ABLATIONS: "3. Benchmarks, Ablations & Quantitative Teardown",
    AnalysisModule.BOTTLENECKS_AND_CRITIQUE: "4. Bottlenecks, Failure Modes & Architectural Critique",
}


MODULE_PROMPTS: dict[AnalysisModule, str] = {
    AnalysisModule.EXECUTIVE_SUMMARY: PROMPT_EXECUTIVE_SUMMARY,
    AnalysisModule.TOPICS_AND_MATH: PROMPT_TOPICS_AND_MATH,
    AnalysisModule.BENCHMARKS_AND_ABLATIONS: PROMPT_BENCHMARKS_AND_ABLATIONS,
    AnalysisModule.BOTTLENECKS_AND_CRITIQUE: PROMPT_BOTTLENECKS_AND_CRITIQUE,
}


# Section-Routed Map-Reduce routing table:
# Maps each analysis module to ordered priority sections from ParsedPaper.sections
SECTION_ROUTING_MAP: dict[AnalysisModule, list[str]] = {
    AnalysisModule.EXECUTIVE_SUMMARY: [
        "Abstract",
        "Introduction",
        "Conclusion",
    ],
    AnalysisModule.TOPICS_AND_MATH: [
        "Abstract",
        "Methodology",
        "Introduction",
        "Appendix",
    ],
    AnalysisModule.BENCHMARKS_AND_ABLATIONS: [
        "Experiments & Benchmarks",
        "Abstract",
        "Conclusion",
    ],
    AnalysisModule.BOTTLENECKS_AND_CRITIQUE: [
        "Methodology",
        "Experiments & Benchmarks",
        "Limitations",
        "Conclusion",
    ],
}


class PaperAnalyzer:
    """
    Coordinates context assembly, token budget protection, streaming execution,
    grounded Q&A, and Markdown/JSON report exports.
    """

    _TRIM_MARKER = (
        "\n\n... [ Middle section trimmed to fit configured VRAM / context window ] ...\n\n"
    )
    _PROMPT_FRAMING_TOKENS = 120
    _MESSAGE_WRAPPER_TOKENS = 12

    def __init__(self, engine: HybridLLMEngine, paper: ParsedPaper) -> None:
        self.engine = engine
        self.paper = paper
        self.effective_strategy = self.engine.resolve_execution_strategy(
            paper.total_tokens_cleaned
        )

    @staticmethod
    def extract_query_terms(question: str) -> set[str]:
        """
        Extracts normalized retrieval terms from a user question while preserving
        hyphenated/underscored technical terms (e.g. KV-cache, fine-tuning,
        Flash-Attention-2) and stripping punctuation.
        """
        raw_tokens = _QUERY_TOKEN_RE.findall((question or "").lower())
        terms: set[str] = set()
        for tok in raw_tokens:
            if tok in _CHAT_STOPWORDS:
                continue
            if "-" in tok or "_" in tok:
                terms.add(tok)
                for part in re.split(r"[-_]+", tok):
                    if len(part) >= 4 and part not in _CHAT_STOPWORDS:
                        terms.add(part)
            elif len(tok) >= 3:
                terms.add(tok)
        return terms

    def _estimate_module_prompt_overhead(self, module: AnalysisModule) -> int:
        """Estimates fixed token overhead for system prompt, task prompt, and framing."""
        return (
            estimate_tokens(BASE_SYSTEM_PROMPT)
            + estimate_tokens(MODULE_PROMPTS[module])
            + estimate_tokens(self.paper.title)
            + self._PROMPT_FRAMING_TOKENS
        )

    def _compute_input_budget(self, prompt_overhead: int) -> tuple[int, int]:
        """
        Computes `(available_context_tokens, effective_max_output_tokens)` guaranteed
        to fit inside `num_ctx`:
            prompt_overhead + available_context_tokens + effective_max_output_tokens <= num_ctx
        Prevents upward clamping when `num_ctx <= max_output_tokens + prompt_overhead`.
        """
        num_ctx = max(0, int(self.engine.config.num_ctx))
        requested_output = max(0, int(self.engine.config.max_output_tokens))
        remaining = max(0, num_ctx - max(0, prompt_overhead))

        if remaining == 0:
            return 0, 0

        if remaining >= requested_output + 512:
            effective_output = requested_output
            available_context = remaining - effective_output
        else:
            effective_output = min(requested_output, max(1, remaining // 2))
            available_context = max(0, remaining - effective_output)

        return available_context, effective_output

    def _fit_to_token_budget(self, text: str, max_tokens: int) -> str:
        """
        Ensures assembled context never exceeds `max_tokens`.
        Preserves both the beginning and end of the text when trimming larger
        segments, and strictly enforces small budgets (<= 500 tokens) as well.
        """
        if max_tokens <= 0 or not text:
            return ""

        current_tokens = estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text

        marker_tokens = estimate_tokens(self._TRIM_MARKER)

        def _shrink_head(source: str, limit: int) -> str:
            char_limit = max(1, int(len(source) * (limit / max(1, estimate_tokens(source))) * 0.9))
            trimmed = source[:char_limit]
            while trimmed and estimate_tokens(trimmed) > limit:
                next_len = int(len(trimmed) * 0.85)
                if next_len >= len(trimmed):
                    next_len = len(trimmed) - 1
                trimmed = trimmed[: max(0, next_len)]
            return trimmed

        if max_tokens <= marker_tokens + 20:
            return _shrink_head(text, max_tokens)

        usable_tokens = max_tokens - marker_tokens
        ratio = usable_tokens / max(1, current_tokens)
        char_budget = max(1, int(len(text) * ratio * 0.92))

        for _ in range(8):
            head_chars = int(char_budget * 0.75)
            tail_chars = max(0, char_budget - head_chars)
            candidate = (
                text[:head_chars]
                + self._TRIM_MARKER
                + (text[-tail_chars:] if tail_chars > 0 else "")
            )
            if estimate_tokens(candidate) <= max_tokens:
                return candidate
            next_budget = int(char_budget * 0.85)
            if next_budget >= char_budget:
                next_budget = char_budget - 1
            char_budget = next_budget
            if char_budget <= 0:
                break

        return _shrink_head(text, max_tokens)

    def build_routed_context(
        self, module: AnalysisModule
    ) -> tuple[str, str, int, list[str]]:
        """
        Returns `(context_markdown, tables_markdown, context_token_count, routed_section_names)`
        according to either Full-Context or Section-Routed Map-Reduce strategy,
        strictly bounded by `num_ctx` minus prompt overhead and output reserve.
        """
        prompt_overhead = self._estimate_module_prompt_overhead(module)
        available_input_tokens, _ = self._compute_input_budget(prompt_overhead)
        tables_share = 0.35 if module == AnalysisModule.BENCHMARKS_AND_ABLATIONS else 0.25

        if self.effective_strategy == ExecutionStrategy.FULL_CONTEXT:
            raw_tables_md = self.paper.get_tables_markdown()
            tables_md = self._fit_to_token_budget(
                raw_tables_md, int(available_input_tokens * tables_share)
            )
            tables_tokens = estimate_tokens(tables_md) if tables_md else 0
            body_budget = max(0, available_input_tokens - tables_tokens)
            fitted_body = self._fit_to_token_budget(
                self.paper.cleaned_markdown, body_budget
            )
            used_sections = list(self.paper.sections.keys())
            total_toks = estimate_tokens(fitted_body) + tables_tokens
            return fitted_body, tables_md, total_toks, used_sections

        # Section-Routed Map-Reduce mode
        target_sections = SECTION_ROUTING_MAP[module]
        assembled_blocks: list[str] = []
        used_sections: list[str] = []

        for sec_name in target_sections:
            sec_content = self.paper.sections.get(sec_name, "").strip()
            if sec_content:
                assembled_blocks.append(f"## [{sec_name}]\n\n{sec_content}")
                used_sections.append(sec_name)

        # Fallback: if none of the target sections were detected, include all available sections
        if not assembled_blocks:
            for sec_name, sec_content in self.paper.sections.items():
                if sec_content.strip():
                    assembled_blocks.append(f"## [{sec_name}]\n\n{sec_content}")
                    used_sections.append(sec_name)

        include_tables = module in (
            AnalysisModule.BENCHMARKS_AND_ABLATIONS,
            AnalysisModule.EXECUTIVE_SUMMARY,
            AnalysisModule.BOTTLENECKS_AND_CRITIQUE,
        )
        raw_tables_md = self.paper.get_tables_markdown() if include_tables else ""
        tables_md = (
            self._fit_to_token_budget(
                raw_tables_md, int(available_input_tokens * tables_share)
            )
            if raw_tables_md
            else ""
        )
        tables_tokens = estimate_tokens(tables_md) if tables_md else 0

        body_budget = max(0, available_input_tokens - tables_tokens)
        combined_sections = "\n\n---\n\n".join(assembled_blocks)
        fitted_body = self._fit_to_token_budget(combined_sections, body_budget)
        total_toks = estimate_tokens(fitted_body) + tables_tokens

        return fitted_body, tables_md, total_toks, used_sections

    def stream_module_analysis(
        self, module: AnalysisModule
    ) -> tuple[Generator[str, None, None], dict[str, object]]:
        """
        Prepares the prompt for a specific analysis module and returns the live
        token generator along with routing telemetry metadata.
        """
        context_md, tables_md, prompt_tokens, routed_sections = self.build_routed_context(
            module
        )
        prompt_overhead = self._estimate_module_prompt_overhead(module)
        _, effective_output = self._compute_input_budget(prompt_overhead)

        exec_note = (
            f"Execution Mode: {self.effective_strategy.value} | "
            f"Routed Sections: {', '.join(routed_sections) if routed_sections else 'Full Document'} | "
            f"Context Tokens: ~{prompt_tokens:,}"
        )

        user_prompt = build_module_user_prompt(
            paper_title=self.paper.title,
            task_instructions=MODULE_PROMPTS[module],
            context_text=context_md,
            tables_markdown=tables_md,
            execution_note=exec_note,
        )

        telemetry: dict[str, object] = {
            "module": module.value,
            "strategy": self.effective_strategy.value,
            "routed_sections": routed_sections,
            "context_tokens": prompt_tokens,
            "model": self.engine.config.model_name,
        }

        stream_kwargs: dict[str, object] = {
            "system_prompt": BASE_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        }
        if 0 < effective_output < self.engine.config.max_output_tokens:
            stream_kwargs["max_tokens"] = effective_output

        stream_gen = self.engine.stream_completion(**stream_kwargs)
        return stream_gen, telemetry

    def _select_bounded_chat_history(
        self,
        chat_history: list[dict[str, str]],
        history_budget: int,
    ) -> tuple[list[dict[str, str]], int]:
        """
        Retains up to the 6 most recent chat turns that fit within `history_budget`
        and returns `(retained_history, estimated_history_tokens)`.
        """
        if not chat_history or history_budget <= 0:
            return [], 0

        retained_reversed: list[dict[str, str]] = []
        used_tokens = 0
        for msg in reversed(chat_history[-6:]):
            msg_tokens = (
                estimate_tokens(msg.get("content", "")) + self._MESSAGE_WRAPPER_TOKENS
            )
            if used_tokens + msg_tokens <= history_budget:
                retained_reversed.append(msg)
                used_tokens += msg_tokens
            else:
                break

        retained_reversed.reverse()
        return retained_reversed, used_tokens

    def stream_paper_chat(
        self,
        question: str,
        chat_history: list[dict[str, str]],
    ) -> tuple[Generator[str, None, None], list[str]]:
        """
        Context-grounded Q&A over the parsed sections and extracted tables.
        Accounts for system prompt, question, and chat history tokens within `num_ctx`,
        and selects the most relevant sections using hyphen-preserving tokenization.
        """
        base_overhead = (
            estimate_tokens(PROMPT_CHAT_SYSTEM)
            + estimate_tokens(question)
            + estimate_tokens(self.paper.title)
            + self._PROMPT_FRAMING_TOKENS
        )
        total_input_budget, effective_output = self._compute_input_budget(base_overhead)

        # Reserve up to 35% of input budget for recent chat history, subtracting exact cost
        history_budget = int(total_input_budget * 0.35)
        recent_history, history_tokens = self._select_bounded_chat_history(
            chat_history, history_budget
        )
        available_paper_tokens = max(0, total_input_budget - history_tokens)

        tables_md = self._fit_to_token_budget(
            self.paper.get_tables_markdown(), int(available_paper_tokens * 0.25)
        )
        tables_tokens = estimate_tokens(tables_md) if tables_md else 0
        section_budget = max(0, available_paper_tokens - tables_tokens)

        if self.effective_strategy == ExecutionStrategy.FULL_CONTEXT:
            selected_sections = list(self.paper.sections.keys())
            context_body = self._fit_to_token_budget(
                self.paper.cleaned_markdown, section_budget
            )
        else:
            # Rank sections by keyword overlap (preserving hyphenated terms) + intent cues
            query_terms = self.extract_query_terms(question)
            question_lower = question.lower()
            scored_sections: list[tuple[int, str, str]] = []
            for sec_name, sec_text in self.paper.sections.items():
                text_lower = sec_text.lower()
                score = sum(
                    text_lower.count(term) * (2 if "-" in term else 1)
                    for term in query_terms
                )
                if sec_name == "Abstract":
                    score += 50
                if (
                    any(
                        k in question_lower
                        for k in ("table", "benchmark", "score", "sota", "ablation", "metric")
                    )
                    and sec_name == "Experiments & Benchmarks"
                ):
                    score += 80
                if (
                    any(
                        k in question_lower
                        for k in ("equation", "loss", "math", "architecture", "method", "theorem")
                    )
                    and sec_name == "Methodology"
                ):
                    score += 80
                if (
                    any(
                        k in question_lower
                        for k in ("limitation", "bottleneck", "fail", "memory", "speed")
                    )
                    and sec_name in ("Limitations", "Methodology")
                ):
                    score += 60
                scored_sections.append((score, sec_name, sec_text))

            scored_sections.sort(key=lambda item: item[0], reverse=True)

            chosen_blocks: list[str] = []
            selected_sections = []
            running_tokens = 0
            header_overhead = 10

            for _, sec_name, sec_text in scored_sections:
                remaining_sec_budget = section_budget - running_tokens
                if remaining_sec_budget <= header_overhead:
                    break
                sec_toks = estimate_tokens(sec_text)
                if sec_toks + header_overhead <= remaining_sec_budget:
                    chosen_blocks.append(f"## [{sec_name}]\n\n{sec_text}")
                    selected_sections.append(sec_name)
                    running_tokens += sec_toks + header_overhead
                elif not chosen_blocks or remaining_sec_budget >= 200:
                    fitted = self._fit_to_token_budget(
                        sec_text, max(0, remaining_sec_budget - header_overhead)
                    )
                    if fitted:
                        chosen_blocks.append(f"## [{sec_name}]\n\n{fitted}")
                        selected_sections.append(sec_name)
                        running_tokens += estimate_tokens(fitted) + header_overhead

            context_body = "\n\n---\n\n".join(chosen_blocks)

        grounded_user_prompt = (
            f"# PAPER TITLE: {self.paper.title}\n\n"
            f"## GROUNDING PAPER SECTIONS ({', '.join(selected_sections)})\n"
            f"{context_body}\n\n"
            f"## EXTRACTED TABLES\n{tables_md}\n\n"
            f"---\n## USER QUESTION\n{question}"
        )

        stream_kwargs: dict[str, object] = {
            "system_prompt": PROMPT_CHAT_SYSTEM,
            "user_prompt": grounded_user_prompt,
            "conversation_history": recent_history,
        }
        if 0 < effective_output < self.engine.config.max_output_tokens:
            stream_kwargs["max_tokens"] = effective_output

        stream = self.engine.stream_completion(**stream_kwargs)
        return stream, selected_sections

    @staticmethod
    def export_to_markdown(
        paper: ParsedPaper,
        results: dict[str, str],
        model_name: str,
        strategy_name: str,
    ) -> str:
        """Generates a complete, publication-grade Markdown technical report."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# Deep Technical Teardown: {paper.title}",
            "",
            f"- **Analyzed Model:** `{model_name}`",
            f"- **Execution Strategy:** `{strategy_name}`",
            f"- **Pages:** {paper.page_count} | **Cleaned Tokens:** {paper.total_tokens_cleaned:,} (Saved {paper.reference_tokens_saved:,} reference tokens)",
            f"- **Detected Sections:** {', '.join(paper.sections.keys())}",
            f"- **Extracted Benchmark Tables:** {len(paper.tables)}",
            f"- **Generated Timestamp:** {timestamp}",
            "",
            "---",
            "",
        ]

        for module in AnalysisModule:
            title = MODULE_TITLES[module]
            body = results.get(module.value, "*Module not yet executed.*")
            lines.append(f"## {title}\n\n{body}\n\n---\n")

        if paper.tables:
            lines.append("## Appendix: Explicitly Extracted Benchmark Tables\n")
            lines.append(paper.get_tables_markdown())

        return "\n".join(lines)

    @staticmethod
    def export_to_json(
        paper: ParsedPaper,
        results: dict[str, str],
        model_name: str,
        strategy_name: str,
    ) -> str:
        """Generates a structured JSON payload of paper metadata, tables, and analysis modules."""
        payload = {
            "metadata": {
                "title": paper.title,
                "authors_or_header": paper.authors_or_header,
                "page_count": paper.page_count,
                "total_tokens_raw": paper.total_tokens_raw,
                "total_tokens_cleaned": paper.total_tokens_cleaned,
                "reference_tokens_saved": paper.reference_tokens_saved,
                "detected_sections": list(paper.sections.keys()),
                "extracted_table_count": len(paper.tables),
                "analyzed_with_model": model_name,
                "execution_strategy": strategy_name,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "analysis_modules": {
                module.value: results.get(module.value, "")
                for module in AnalysisModule
            },
            "extracted_tables": [
                {
                    "table_id": t.table_id,
                    "page_number": t.page_number,
                    "caption": t.caption,
                    "row_count": t.row_count,
                    "col_count": t.col_count,
                    "markdown": t.markdown,
                }
                for t in paper.tables
            ],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)
