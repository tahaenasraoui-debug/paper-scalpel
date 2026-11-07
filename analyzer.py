"""
analyzer.py
Orchestrates the 4 deep technical teardown modules and paper Q&A.
Implements both Full-Context execution and VRAM-safe Section-Routed Map-Reduce.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Generator, List, Optional, Tuple

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


class AnalysisModule(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    TOPICS_AND_MATH = "topics_and_math"
    BENCHMARKS_AND_ABLATIONS = "benchmarks_and_ablations"
    BOTTLENECKS_AND_CRITIQUE = "bottlenecks_and_critique"


MODULE_TITLES: Dict[AnalysisModule, str] = {
    AnalysisModule.EXECUTIVE_SUMMARY: "1. Executive Summary & Core Delta",
    AnalysisModule.TOPICS_AND_MATH: "2. Important Topics, Taxonomy & Math Formulation",
    AnalysisModule.BENCHMARKS_AND_ABLATIONS: "3. Benchmarks, Ablations & Quantitative Teardown",
    AnalysisModule.BOTTLENECKS_AND_CRITIQUE: "4. Bottlenecks, Failure Modes & Architectural Critique",
}


MODULE_PROMPTS: Dict[AnalysisModule, str] = {
    AnalysisModule.EXECUTIVE_SUMMARY: PROMPT_EXECUTIVE_SUMMARY,
    AnalysisModule.TOPICS_AND_MATH: PROMPT_TOPICS_AND_MATH,
    AnalysisModule.BENCHMARKS_AND_ABLATIONS: PROMPT_BENCHMARKS_AND_ABLATIONS,
    AnalysisModule.BOTTLENECKS_AND_CRITIQUE: PROMPT_BOTTLENECKS_AND_CRITIQUE,
}


# Section-Routed Map-Reduce routing table:
# Maps each analysis module to ordered priority sections from ParsedPaper.sections
SECTION_ROUTING_MAP: Dict[AnalysisModule, List[str]] = {
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

    def __init__(self, engine: HybridLLMEngine, paper: ParsedPaper) -> None:
        self.engine = engine
        self.paper = paper
        self.effective_strategy = self.engine.resolve_execution_strategy(
            paper.total_tokens_cleaned
        )

    def _fit_to_token_budget(self, text: str, max_tokens: int) -> str:
        """
        Ensures assembled section context never overflows the model's context window.
        Preserves both the beginning and end of the text if trimming is required.
        """
        current_tokens = estimate_tokens(text)
        if current_tokens <= max_tokens or max_tokens <= 500:
            return text

        ratio = max_tokens / max(1, current_tokens)
        char_budget = int(len(text) * ratio * 0.92)
        head_chars = int(char_budget * 0.75)
        tail_chars = max(0, char_budget - head_chars)

        return (
            text[:head_chars]
            + "\n\n... [ Middle section trimmed to fit configured VRAM / context window ] ...\n\n"
            + (text[-tail_chars:] if tail_chars > 0 else "")
        )

    def build_routed_context(
        self, module: AnalysisModule
    ) -> Tuple[str, str, int, List[str]]:
        """
        Returns `(context_markdown, tables_markdown, context_token_count, routed_section_names)`
        according to either Full-Context or Section-Routed Map-Reduce strategy.
        """
        # Reserve room for system prompt, instructions, and max_output_tokens
        available_input_tokens = max(
            2048,
            int(self.engine.config.num_ctx - self.engine.config.max_output_tokens - 900),
        )

        if self.effective_strategy == ExecutionStrategy.FULL_CONTEXT:
            tables_md = self.paper.get_tables_markdown()
            tables_tokens = estimate_tokens(tables_md)
            body_budget = max(1500, available_input_tokens - tables_tokens)
            fitted_body = self._fit_to_token_budget(
                self.paper.cleaned_markdown, body_budget
            )
            used_sections = list(self.paper.sections.keys())
            total_toks = estimate_tokens(fitted_body) + tables_tokens
            return fitted_body, tables_md, total_toks, used_sections

        # Section-Routed Map-Reduce mode
        target_sections = SECTION_ROUTING_MAP[module]
        assembled_blocks: List[str] = []
        used_sections: List[str] = []

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
        tables_md = self.paper.get_tables_markdown() if include_tables else ""
        tables_tokens = estimate_tokens(tables_md) if tables_md else 0

        body_budget = max(1200, available_input_tokens - tables_tokens)
        combined_sections = "\n\n---\n\n".join(assembled_blocks)
        fitted_body = self._fit_to_token_budget(combined_sections, body_budget)
        total_toks = estimate_tokens(fitted_body) + tables_tokens

        return fitted_body, tables_md, total_toks, used_sections

    def stream_module_analysis(
        self, module: AnalysisModule
    ) -> Tuple[Generator[str, None, None], Dict[str, object]]:
        """
        Prepares the prompt for a specific analysis module and returns the live
        token generator along with routing telemetry metadata.
        """
        context_md, tables_md, prompt_tokens, routed_sections = self.build_routed_context(
            module
        )

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

        telemetry = {
            "module": module.value,
            "strategy": self.effective_strategy.value,
            "routed_sections": routed_sections,
            "context_tokens": prompt_tokens,
            "model": self.engine.config.model_name,
        }

        stream_gen = self.engine.stream_completion(
            system_prompt=BASE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return stream_gen, telemetry

    def stream_paper_chat(
        self,
        question: str,
        chat_history: List[Dict[str, str]],
    ) -> Tuple[Generator[str, None, None], List[str]]:
        """
        Context-grounded Q&A over the parsed sections and extracted tables.
        Selects the most relevant sections based on keyword overlap when in
        Section-Routed mode, or passes full context when in Full-Context mode.
        """
        available_input_tokens = max(
            2048,
            int(self.engine.config.num_ctx - self.engine.config.max_output_tokens - 1200),
        )

        if self.effective_strategy == ExecutionStrategy.FULL_CONTEXT:
            selected_sections = list(self.paper.sections.keys())
            context_body = self._fit_to_token_budget(
                self.paper.cleaned_markdown, int(available_input_tokens * 0.8)
            )
        else:
            # Rank sections by keyword overlap with the user's query + always include Abstract
            query_terms = {
                w.lower()
                for w in question.split()
                if len(w) > 3 and w.isalnum()
            }
            scored_sections: List[Tuple[int, str, str]] = []
            for sec_name, sec_text in self.paper.sections.items():
                text_lower = sec_text.lower()
                score = sum(text_lower.count(term) for term in query_terms)
                if sec_name == "Abstract":
                    score += 50
                if any(k in question.lower() for k in ("table", "benchmark", "score", "sota", "ablation", "metric")) and sec_name == "Experiments & Benchmarks":
                    score += 80
                if any(k in question.lower() for k in ("equation", "loss", "math", "architecture", "method", "theorem")) and sec_name == "Methodology":
                    score += 80
                if any(k in question.lower() for k in ("limitation", "bottleneck", "fail", "memory", "speed")) and sec_name in ("Limitations", "Methodology"):
                    score += 60
                scored_sections.append((score, sec_name, sec_text))

            scored_sections.sort(key=lambda item: item[0], reverse=True)

            chosen_blocks: List[str] = []
            selected_sections = []
            running_tokens = 0
            section_budget = int(available_input_tokens * 0.75)

            for _, sec_name, sec_text in scored_sections:
                sec_toks = estimate_tokens(sec_text)
                if running_tokens + sec_toks <= section_budget or not chosen_blocks:
                    fitted = self._fit_to_token_budget(
                        sec_text, max(1000, section_budget - running_tokens)
                    )
                    chosen_blocks.append(f"## [{sec_name}]\n\n{fitted}")
                    selected_sections.append(sec_name)
                    running_tokens += estimate_tokens(fitted)

            context_body = "\n\n---\n\n".join(chosen_blocks)

        tables_md = self._fit_to_token_budget(
            self.paper.get_tables_markdown(), int(available_input_tokens * 0.25)
        )

        grounded_user_prompt = (
            f"# PAPER TITLE: {self.paper.title}\n\n"
            f"## GROUNDING PAPER SECTIONS ({', '.join(selected_sections)})\n"
            f"{context_body}\n\n"
            f"## EXTRACTED TABLES\n{tables_md}\n\n"
            f"---\n## USER QUESTION\n{question}"
        )

        # Keep recent 6 messages of history to preserve context window
        recent_history = chat_history[-6:] if chat_history else []
        stream = self.engine.stream_completion(
            system_prompt=PROMPT_CHAT_SYSTEM,
            user_prompt=grounded_user_prompt,
            conversation_history=recent_history,
        )
        return stream, selected_sections

    @staticmethod
    def export_to_markdown(
        paper: ParsedPaper,
        results: Dict[str, str],
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
        results: Dict[str, str],
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
