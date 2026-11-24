from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from analyzer import AnalysisModule, PaperAnalyzer
from llm_engine import ExecutionStrategy, HybridLLMEngine, LLMConfig, ProviderMode
from parser import ExtractedTable, ParsedPaper, estimate_tokens


def _build_sample_paper(scale: int = 1) -> ParsedPaper:
    abstract = ("Abstract summary about sparse routing. " * (15 * scale)).strip()
    intro = ("Introduction covering motivation and baselines. " * (40 * scale)).strip()
    methodology = (
        "Methodology introducing KV-cache eviction, Flash-Attention-2 tiling, "
        "and zero-shot fine-tuning equations. " * (50 * scale)
    ).strip()
    experiments = (
        "Experiments evaluating MMLU and GSM8K benchmarks with ablation tables. " * (50 * scale)
    ).strip()
    limitations = (
        "Limitations include VRAM spikes at extreme sequence lengths. " * (20 * scale)
    ).strip()
    conclusion = ("Conclusion summarizing practical trade-offs. " * (15 * scale)).strip()

    sections = {
        "Abstract": abstract,
        "Introduction": intro,
        "Methodology": methodology,
        "Experiments & Benchmarks": experiments,
        "Limitations": limitations,
        "Conclusion": conclusion,
    }
    cleaned_md = "\n\n---\n\n".join(f"## [{k}]\n\n{v}" for k, v in sections.items())
    table_md = (
        "| Model | MMLU | GSM8K |\n"
        "| --- | --- | --- |\n"
        "| Baseline | 71.0 | 64.2 |\n"
        "| Proposed | 76.5 | 72.8 |"
    )
    tables = [
        ExtractedTable(
            table_id="Table-1",
            page_number=4,
            caption="Table 1: Benchmark Scores",
            markdown=table_md,
            row_count=3,
            col_count=3,
        )
    ]
    cleaned_toks = estimate_tokens(cleaned_md)
    return ParsedPaper(
        title="ScalpelFormer: Sparse Routing / Fast KV-Cache",
        authors_or_header="Doe et al.",
        page_count=12,
        raw_markdown=cleaned_md + "\n\nReferences\n[1] Prior work.",
        cleaned_markdown=cleaned_md,
        sections=sections,
        references_text="[1] Prior work.",
        tables=tables,
        total_tokens_raw=cleaned_toks + 50,
        total_tokens_cleaned=cleaned_toks,
        reference_tokens_saved=50,
    )


def _make_engine(
    num_ctx: int = 16384,
    max_output_tokens: int = 3072,
    strategy: ExecutionStrategy = ExecutionStrategy.SECTION_ROUTED,
) -> HybridLLMEngine:
    cfg = LLMConfig(
        provider_mode=ProviderMode.LOCAL_OLLAMA,
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:7b-instruct",
        num_ctx=num_ctx,
        max_output_tokens=max_output_tokens,
        execution_strategy=strategy,
    )
    with patch("llm_engine.OpenAI"):
        return HybridLLMEngine(cfg)


def test_fit_to_token_budget_enforces_small_and_zero_budgets() -> None:
    paper = _build_sample_paper(scale=3)
    analyzer = PaperAnalyzer(_make_engine(), paper)

    long_text = "technical evaluation benchmark equation " * 500
    assert analyzer._fit_to_token_budget(long_text, 0) == ""

    # Regression: previously budgets <= 500 returned untrimmed text
    for budget in (40, 200, 450, 800):
        fitted = analyzer._fit_to_token_budget(long_text, budget)
        assert fitted
        assert estimate_tokens(fitted) <= budget


def test_token_budget_clamping_when_num_ctx_le_max_output_plus_900() -> None:
    paper = _build_sample_paper(scale=6)

    # Case 1: num_ctx == max_output_tokens (e.g. 8192 and 8192)
    engine_equal = _make_engine(num_ctx=8192, max_output_tokens=8192)
    analyzer_equal = PaperAnalyzer(engine_equal, paper)
    for mod in AnalysisModule:
        body, tables_md, ctx_toks, _ = analyzer_equal.build_routed_context(mod)
        overhead = analyzer_equal._estimate_module_prompt_overhead(mod)
        assert ctx_toks == estimate_tokens(body) + (estimate_tokens(tables_md) if tables_md else 0)
        assert ctx_toks + overhead < 8192

    # Case 2: num_ctx smaller than max_output_tokens + 900 (2048 vs 3072)
    engine_tight = _make_engine(num_ctx=2048, max_output_tokens=3072)
    analyzer_tight = PaperAnalyzer(engine_tight, paper)
    overhead = analyzer_tight._estimate_module_prompt_overhead(AnalysisModule.EXECUTIVE_SUMMARY)
    _, _, ctx_toks, _ = analyzer_tight.build_routed_context(AnalysisModule.EXECUTIVE_SUMMARY)
    # Must NEVER clamp upward to 2048 and overflow num_ctx
    assert ctx_toks + overhead < 2048

    # Verify stream_module_analysis clamps max_tokens when window is tight
    engine_tight.stream_completion = MagicMock(return_value=iter(["ok"]))
    stream_gen, _ = analyzer_tight.stream_module_analysis(AnalysisModule.EXECUTIVE_SUMMARY)
    assert list(stream_gen) == ["ok"]
    call_kwargs = engine_tight.stream_completion.call_args.kwargs
    assert "max_tokens" in call_kwargs
    assert call_kwargs["max_tokens"] < 3072
    assert ctx_toks + overhead + call_kwargs["max_tokens"] <= 2048


def test_extract_query_terms_preserves_hyphenated_terms_and_strips_punctuation() -> None:
    terms = PaperAnalyzer.extract_query_terms(
        "What is the KV-cache overhead of Flash-Attention-2 and zero-shot fine-tuning?"
    )
    assert "kv-cache" in terms
    assert "flash-attention-2" in terms
    assert "zero-shot" in terms
    assert "fine-tuning" in terms
    assert "overhead" in terms
    assert "what" not in terms


def test_chat_retrieval_ranks_hyphenated_match_and_accounts_for_history_tokens() -> None:
    paper = _build_sample_paper(scale=2)
    # Make a tight budget so only top-ranked sections fit
    engine = _make_engine(
        num_ctx=4096,
        max_output_tokens=2048,
        strategy=ExecutionStrategy.SECTION_ROUTED,
    )
    engine.stream_completion = MagicMock(return_value=iter(["Grounded answer."]))
    analyzer = PaperAnalyzer(engine, paper)

    large_history = [
        {"role": "user", "content": "Prior question " * 120},
        {"role": "assistant", "content": "Prior detailed answer " * 200},
        {"role": "user", "content": "Follow-up " * 120},
        {"role": "assistant", "content": "Follow-up response " * 200},
    ]

    stream_gen, selected_sections = analyzer.stream_paper_chat(
        question="How does KV-cache eviction work with Flash-Attention-2 and fine-tuning?",
        chat_history=large_history,
    )
    assert list(stream_gen) == ["Grounded answer."]
    assert "Methodology" in selected_sections

    call_kwargs = engine.stream_completion.call_args.kwargs
    sent_history = call_kwargs["conversation_history"]
    sent_system = call_kwargs["system_prompt"]
    sent_user = call_kwargs["user_prompt"]

    # History must be bounded so total input + output stays within num_ctx
    total_sent_input = (
        estimate_tokens(sent_system)
        + estimate_tokens(sent_user)
        + sum(estimate_tokens(m["content"]) for m in sent_history)
    )
    effective_out = call_kwargs.get("max_tokens", engine.config.max_output_tokens)
    assert len(sent_history) < len(large_history)
    assert total_sent_input + effective_out <= engine.config.num_ctx


def test_exports_markdown_and_json() -> None:
    paper = _build_sample_paper(scale=1)
    results = {AnalysisModule.EXECUTIVE_SUMMARY.value: "Core delta summary [Sec. 1]."}

    md = PaperAnalyzer.export_to_markdown(
        paper=paper,
        results=results,
        model_name="qwen2.5:7b-instruct",
        strategy_name=ExecutionStrategy.SECTION_ROUTED.value,
    )
    assert paper.title in md
    assert "Core delta summary [Sec. 1]." in md
    assert "Table-1" in md

    raw_json = PaperAnalyzer.export_to_json(
        paper=paper,
        results=results,
        model_name="qwen2.5:7b-instruct",
        strategy_name=ExecutionStrategy.SECTION_ROUTED.value,
    )
    parsed = json.loads(raw_json)
    assert parsed["metadata"]["title"] == paper.title
    assert parsed["analysis_modules"]["executive_summary"] == "Core delta summary [Sec. 1]."
    assert parsed["metadata"]["extracted_table_count"] == 1
