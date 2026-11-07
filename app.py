"""
app.py
Interactive Streamlit Web Application for the Research Paper Analyzer.
Supports Local AI (Ollama / llama.cpp) and Cloud LLM APIs with real-time
token streaming across 5 specialized tabs and 1-click Markdown/JSON export.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

from analyzer import MODULE_TITLES, AnalysisModule, PaperAnalyzer
from llm_engine import (
    PROVIDER_DEFAULTS,
    ExecutionStrategy,
    HybridLLMEngine,
    LLMConfig,
    ProviderMode,
)
from parser import AcademicPaperParser, ParsedPaper

load_dotenv()

st.set_page_config(
    page_title="Research Paper Deep Teardown Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Initializes persistent Streamlit session state keys."""
    defaults = {
        "parsed_paper": None,
        "paper_hash": None,
        "analysis_results": {},
        "module_telemetry": {},
        "chat_messages": [],
        "endpoint_models": [],
        "endpoint_status": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.cache_data(show_spinner=False)
def cached_parse_pdf(pdf_bytes: bytes, strip_references: bool) -> ParsedPaper:
    """Caches layout-aware PDF parsing and table extraction across UI reruns."""
    parser = AcademicPaperParser()
    return parser.parse_pdf(pdf_bytes, strip_references=strip_references)


def render_sidebar() -> LLMConfig:
    """Renders the Hybrid Local + Cloud LLM configuration sidebar."""
    with st.sidebar:
        st.title("⚙️ Inference Engine")
        st.caption("Switch seamlessly between Local VRAM & Cloud Frontier APIs")

        backend_category = st.radio(
            "Inference Backend",
            options=["Local AI (Ollama / llama.cpp)", "Cloud API (Kimi / OpenAI / Gemini / Claude)"],
            index=0,
        )

        if backend_category.startswith("Local AI"):
            provider_options = [
                ProviderMode.LOCAL_OLLAMA,
                ProviderMode.LOCAL_LLAMACPP,
                ProviderMode.CUSTOM_ENDPOINT,
            ]
        else:
            provider_options = [
                ProviderMode.CLOUD_MOONSHOT,
                ProviderMode.CLOUD_OPENAI,
                ProviderMode.CLOUD_GEMINI,
                ProviderMode.CLOUD_ANTHROPIC,
                ProviderMode.CUSTOM_ENDPOINT,
            ]

        selected_provider = st.selectbox(
            "Provider Preset",
            options=provider_options,
            format_func=lambda p: p.value,
        )

        defaults = PROVIDER_DEFAULTS[selected_provider]
        base_url = st.text_input(
            "Base URL (OpenAI-Compatible)",
            value=os.getenv("LLM_BASE_URL", str(defaults["base_url"])),
        )

        # Hide API Key in Local Mode unless Custom Endpoint is chosen
        if backend_category.startswith("Local AI") and selected_provider != ProviderMode.CUSTOM_ENDPOINT:
            api_key = "local-no-key-required"
            st.caption("🔒 **Local Mode Active:** No API key required; data stays on-device.")
        else:
            env_key = (
                os.getenv("MOONSHOT_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("LLM_API_KEY")
                or ""
            )
            api_key = st.text_input(
                "API Key",
                value=env_key,
                type="password",
                help="Required for Cloud APIs (Moonshot/Kimi, OpenAI, Gemini, Anthropic).",
            )

        # Optional button to query live models from local Ollama/llama.cpp or Cloud endpoint
        col_ping, col_hint = st.columns([1, 1])
        with col_ping:
            if st.button("🔍 Detect Models", use_container_width=True):
                ok, models, msg = HybridLLMEngine.check_endpoint_and_list_models(
                    base_url=base_url, api_key=api_key
                )
                st.session_state["endpoint_models"] = models if ok else []
                st.session_state["endpoint_status"] = msg

        if st.session_state["endpoint_status"]:
            st.caption(st.session_state["endpoint_status"])

        if st.session_state["endpoint_models"]:
            model_name = st.selectbox(
                "Detected Models",
                options=st.session_state["endpoint_models"],
            )
        else:
            model_name = st.text_input(
                "Model Name",
                value=os.getenv("LLM_MODEL_NAME", str(defaults["default_model"])),
                help="Examples: qwen2.5:7b-instruct, llama3.1:8b, deepseek-r1:8b, kimi-k2-0711-preview, gpt-4o",
            )

        st.divider()
        st.subheader("🧠 Execution & VRAM Strategy")

        execution_strategy = st.selectbox(
            "Execution Mode",
            options=[
                ExecutionStrategy.AUTO,
                ExecutionStrategy.FULL_CONTEXT,
                ExecutionStrategy.SECTION_ROUTED,
            ],
            format_func=lambda s: s.value,
            help=(
                "Auto uses Section-Routed Map-Reduce for Local 8k-24k context models "
                "and Full-Context for large-window Cloud APIs."
            ),
        )

        strip_references = st.checkbox(
            "Strip References to Save VRAM (20–30% tokens)",
            value=True,
            help="Automatically strips the Bibliography/References section before LLM ingestion.",
        )

        col_temp, col_ctx = st.columns(2)
        with col_temp:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.05,
            )
        with col_ctx:
            num_ctx = st.select_slider(
                "Context Window (num_ctx)",
                options=[8192, 16384, 32768, 65536, 131072, 200000],
                value=int(defaults["default_ctx"]),
            )

        max_output_tokens = st.slider(
            "Max Output Tokens per Module",
            min_value=1024,
            max_value=8192,
            value=3072,
            step=512,
        )

        return LLMConfig(
            provider_mode=selected_provider,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            num_ctx=num_ctx,
            max_output_tokens=max_output_tokens,
            execution_strategy=execution_strategy,
            strip_references=strip_references,
        )


def render_paper_telemetry(paper: ParsedPaper, resolved_strategy: ExecutionStrategy) -> None:
    """Renders paper metadata metrics and expandable section/table inspectors."""
    st.markdown(f"### 📄 {paper.title}")
    if paper.authors_or_header:
        st.caption(paper.authors_or_header)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pages", f"{paper.page_count}")
    c2.metric(
        "Est. Tokens (Cleaned)",
        f"{paper.total_tokens_cleaned:,}",
        delta=f"-{paper.reference_tokens_saved:,} ref tokens" if paper.reference_tokens_saved > 0 else "0 stripped",
        delta_color="normal",
    )
    c3.metric("Detected Sections", f"{len(paper.sections)}")
    c4.metric("Extracted Tables", f"{len(paper.tables)}")
    c5.metric(
        "Active Strategy",
        "Map-Reduce" if resolved_strategy == ExecutionStrategy.SECTION_ROUTED else "Full-Context",
    )

    with st.expander("🔎 Inspect Parsed Sections & Extracted Benchmark Tables", expanded=False):
        sec_col, tbl_col = st.columns(2)
        with sec_col:
            st.markdown("**Detected Canonical Sections & Token Breakdown**")
            for sec_name, sec_text in paper.sections.items():
                st.markdown(
                    f"- **{sec_name}**: `{len(sec_text):,}` chars (~`{int(len(sec_text)/3.8):,}` tokens)"
                )
            if paper.references_text:
                st.markdown(
                    f"- *Stripped References Isolated*: `{len(paper.references_text):,}` chars"
                )
        with tbl_col:
            st.markdown(f"**Explicitly Extracted Benchmark Tables ({len(paper.tables)})**")
            if not paper.tables:
                st.info("No explicit grid tables found; numeric metrics will be parsed from Markdown body.")
            else:
                for tbl in paper.tables:
                    st.markdown(f"**{tbl.table_id}** (Page {tbl.page_number}): *{tbl.caption}*")
                    st.markdown(tbl.markdown)
                    st.divider()


def run_and_stream_module(
    analyzer: PaperAnalyzer,
    module: AnalysisModule,
    container: st.delta_generator.DeltaGenerator,
) -> None:
    """Streams a single analysis module into the provided Streamlit container."""
    with container:
        try:
            stream_gen, telemetry = analyzer.stream_module_analysis(module)
            st.caption(
                f"⚡ **Routing Telemetry:** Strategy=`{telemetry['strategy']}` | "
                f"Sections=`{', '.join(telemetry['routed_sections'])}` | "
                f"Input Context=`~{telemetry['context_tokens']:,} tokens` | "
                f"Model=`{telemetry['model']}`"
            )
            full_response = st.write_stream(stream_gen)
            st.session_state["analysis_results"][module.value] = str(full_response)
            st.session_state["module_telemetry"][module.value] = telemetry
        except Exception as exc:
            st.error(f"Error during `{MODULE_TITLES[module]}`: {exc}")


def main() -> None:
    init_session_state()
    llm_config = render_sidebar()

    st.title("🔬 Research Paper Deep Technical Analyzer")
    st.markdown(
        "Upload any multi-column academic PDF to extract reading-order LaTeX & benchmark tables, "
        "then run an anti-hallucination architectural teardown via **Local AI (Ollama / llama.cpp)** or **Cloud Frontier APIs**."
    )

    uploaded_file = st.file_uploader(
        "Drag and drop a Research Paper PDF",
        type=["pdf"],
        help="Supports multi-column arXiv, NeurIPS, ICML, ICLR, CVPR, ACL, and IEEE PDFs.",
    )

    if uploaded_file is None:
        st.info(
            "👆 Upload a research paper PDF above to inspect its parsed sections, benchmark tables, and launch the 4-stage technical teardown."
        )
        return

    pdf_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(
        pdf_bytes + str(llm_config.strip_references).encode("utf-8")
    ).hexdigest()

    if st.session_state["paper_hash"] != current_hash:
        with st.spinner("Parsing multi-column PDF layout, LaTeX equations, sections, and benchmark tables..."):
            parsed_paper = cached_parse_pdf(
                pdf_bytes, strip_references=llm_config.strip_references
            )
            st.session_state["parsed_paper"] = parsed_paper
            st.session_state["paper_hash"] = current_hash
            st.session_state["analysis_results"] = {}
            st.session_state["module_telemetry"] = {}
            st.session_state["chat_messages"] = []

    paper: ParsedPaper = st.session_state["parsed_paper"]
    engine = HybridLLMEngine(llm_config)
    analyzer = PaperAnalyzer(engine=engine, paper=paper)

    render_paper_telemetry(paper, analyzer.effective_strategy)

    # Action bar: Run All vs. Export Report
    action_col1, action_col2, action_col3 = st.columns([2, 1, 1])
    with action_col1:
        run_all_clicked = st.button(
            "🚀 Run Full 4-Module Technical Teardown (Live Stream)",
            type="primary",
            use_container_width=True,
        )

    with action_col2:
        md_report = PaperAnalyzer.export_to_markdown(
            paper=paper,
            results=st.session_state["analysis_results"],
            model_name=llm_config.model_name,
            strategy_name=analyzer.effective_strategy.value,
        )
        st.download_button(
            label="📥 Export Report (.md)",
            data=md_report.encode("utf-8"),
            file_name=f"{paper.title[:45].replace(' ', '_')}_teardown.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=len(st.session_state["analysis_results"]) == 0,
        )

    with action_col3:
        json_report = PaperAnalyzer.export_to_json(
            paper=paper,
            results=st.session_state["analysis_results"],
            model_name=llm_config.model_name,
            strategy_name=analyzer.effective_strategy.value,
        )
        st.download_button(
            label="📦 Export Data (.json)",
            data=json_report.encode("utf-8"),
            file_name=f"{paper.title[:45].replace(' ', '_')}_teardown.json",
            mime="application/json",
            use_container_width=True,
            disabled=len(st.session_state["analysis_results"]) == 0,
        )

    # 5 Interactive Workspace Tabs
    tab_summary, tab_math, tab_bench, tab_critique, tab_chat = st.tabs(
        [
            "📋 1. Executive Summary",
            "🧮 2. Topics & Math",
            "📊 3. Benchmarks & Ablations",
            "🛠️ 4. Bottlenecks & Critique",
            "💬 5. Chat with Paper",
        ]
    )

    tab_map = {
        AnalysisModule.EXECUTIVE_SUMMARY: tab_summary,
        AnalysisModule.TOPICS_AND_MATH: tab_math,
        AnalysisModule.BENCHMARKS_AND_ABLATIONS: tab_bench,
        AnalysisModule.BOTTLENECKS_AND_CRITIQUE: tab_critique,
    }

    # If user clicked "Run Full 4-Module Technical Teardown", stream sequentially into each tab
    if run_all_clicked:
        for module, tab_obj in tab_map.items():
            with tab_obj:
                st.subheader(MODULE_TITLES[module])
                stream_box = st.container()
                run_and_stream_module(analyzer, module, stream_box)

    # Render individual module tabs (cached output + individual Re-run button)
    for module, tab_obj in tab_map.items():
        with tab_obj:
            if not run_all_clicked:
                hdr_col, btn_col = st.columns([4, 1])
                with hdr_col:
                    st.subheader(MODULE_TITLES[module])
                with btn_col:
                    rerun_single = st.button(
                        f"▶️ Run Module",
                        key=f"btn_{module.value}",
                        use_container_width=True,
                    )

                stream_box = st.container()
                if rerun_single:
                    run_and_stream_module(analyzer, module, stream_box)
                elif module.value in st.session_state["analysis_results"]:
                    telemetry = st.session_state["module_telemetry"].get(module.value)
                    if telemetry:
                        st.caption(
                            f"⚡ **Routing Telemetry:** Strategy=`{telemetry['strategy']}` | "
                            f"Sections=`{', '.join(telemetry['routed_sections'])}` | "
                            f"Input Context=`~{telemetry['context_tokens']:,} tokens` | "
                            f"Model=`{telemetry['model']}`"
                        )
                    st.markdown(st.session_state["analysis_results"][module.value])
                else:
                    st.info(
                        f"Click **▶️ Run Module** on the right or **🚀 Run Full 4-Module Technical Teardown** above to generate `{MODULE_TITLES[module]}`."
                    )

    # Tab 5: Context-Grounded Paper Chat (RAG / Section-Routed Q&A)
    with tab_chat:
        st.subheader("💬 Context-Grounded Technical Q&A")
        st.caption(
            "Ask specific questions about proofs, hyperparameters, equations, or benchmark anomalies. "
            "Responses are strictly grounded in the parsed paper sections and extracted tables."
        )

        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input(
            "Ask a deep technical question (e.g., 'What is the exact complexity of Eq. 5 and how was Table 3 ablated?')"
        )
        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                try:
                    stream_gen, cited_sections = analyzer.stream_paper_chat(
                        question=user_question,
                        chat_history=st.session_state["chat_messages"],
                    )
                    st.caption(
                        f"📌 *Grounded in sections: {', '.join(cited_sections)} + Extracted Benchmark Tables*"
                    )
                    answer_text = st.write_stream(stream_gen)
                    st.session_state["chat_messages"].append(
                        {"role": "user", "content": user_question}
                    )
                    st.session_state["chat_messages"].append(
                        {"role": "assistant", "content": str(answer_text)}
                    )
                except Exception as exc:
                    st.error(f"Chat inference error: {exc}")


if __name__ == "__main__":
    main()
