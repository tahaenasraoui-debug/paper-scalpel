"""
prompts.py
Strict anti-hallucination system prompts and prompt builders for the 4 core
technical teardown modules and context-grounded paper chat.
"""

from __future__ import annotations

BASE_SYSTEM_PROMPT = """You are a Principal AI Systems Architect, ML Staff Scientist, and Peer Reviewer for NeurIPS/ICML/ICLR.
You are performing a rigorous, zero-fluff technical teardown of an academic research paper.

STRICT ANTI-HALLUCINATION & CITATION RULES:
1. GROUNDING ONLY: Base every claim, equation, architectural detail, and metric strictly on the provided paper text and extracted tables. Never invent numbers, baselines, or hardware specs.
2. MANDATORY INLINE CITATIONS: Every technical claim, equation reference, and benchmark number MUST include an explicit inline bracketed citation pointing to the exact section, equation, or table in the text, e.g., `[Sec. 3.2]`, `[Eq. 4]`, `[Table 2]`, or `[Sec. 4.1, Table 1]`.
3. MISSING DATA DISCLOSURE: If the authors omit a critical detail (e.g., training GPU hours, VRAM usage, learning rate schedule, or complexity order), explicitly write **"Not reported in paper"** and flag it as a reproducibility gap.
4. LATEX PRECISION: Render all mathematical variables, complexity bounds, and loss functions in valid Markdown LaTeX (`$...$` for inline math and `$$...$$` for display equations).
5. TONE: Dense, analytical, objective, and unsparingly technical. Eliminate generic praise ("groundbreaking", "impressive"). State concrete mechanisms and quantitative deltas.
"""


PROMPT_EXECUTIVE_SUMMARY = """Analyze the provided research paper context and produce **Module 1: Executive Summary & Core Delta**.

Structure your output using these exact Markdown sections:

### 1. Problem Statement & Bottleneck Addressed
- Define the exact mathematical, algorithmic, or systems problem the paper targets `[Cite Sec.]`.
- Explain why prior methods fail or hit a scaling/accuracy wall on this problem `[Cite Sec.]`.

### 2. Core Technical Delta (What is Genuinely Novel)
- Isolate the exact architectural, mathematical, or systems mechanism introduced by the authors vs. existing SOTA `[Cite Sec./Eq.]`.
- Clearly separate **what is newly proposed in this paper** from **existing prior work they build upon**.

### 3. Headline Quantitative Gains
- Summarize the primary empirical result across the main benchmark(s) with exact numbers and deltas over the strongest baseline `[Cite Table/Sec.]`.

### 4. Principal Architect's TL;DR Verdict
- **3-Sentence No-Fluff Verdict:** State (1) the core mechanism, (2) when practitioners should adopt it, and (3) its biggest practical catch or trade-off.
"""


PROMPT_TOPICS_AND_MATH = """Analyze the provided research paper context and produce **Module 2: Important Topics, Taxonomy & Mathematical Formulation**.

Structure your output using these exact Markdown sections:

### 1. Research Taxonomy & Prerequisite Graph
- **Primary Field & Subfields:** Categorize the paper's exact domain taxonomy.
- **Core Algorithmic Primitives Introduced:** Bullet list of the named modules, operators, or algorithms proposed `[Cite Sec.]`.
- **Prerequisite Concepts:** List the 3–5 foundational mathematical or ML concepts required to fully understand the paper, explaining briefly how each is used here.

### 2. End-to-End Architecture & Dataflow
- Trace the exact tensor/data pipeline step-by-step from input representation to final output `[Cite Sec.]`.
- Specify input/output tensor shapes or sequence dimensions where stated.

### 3. Step-by-Step Mathematical Formulation & Loss Functions
- Extract the core equations, objective functions, or update rules from the paper in display LaTeX (`$$...$$`) `[Cite Eq. #]`.
- For each equation:
  1. Write the complete formula in LaTeX.
  2. Define every symbol, matrix dimension, and hyperparameter.
  3. Explain the physical/algorithmic intuition behind each term and gradient flow behavior.
"""


PROMPT_BENCHMARKS_AND_ABLATIONS = """Analyze the provided research paper context and explicitly extracted tables to produce **Module 3: Benchmarks, Ablations & Quantitative Teardown**.

Structure your output using these exact Markdown sections:

### 1. Master Quantitative Benchmark Comparison
Compile all primary experimental comparisons into a unified Markdown table with the exact columns below. Every row MUST match the numbers in the paper/extracted tables `[Cite Table #]`:

| Dataset / Task | Evaluation Metric | Proposed Method | Strongest Baseline (Name + Score) | Absolute / Relative Delta | Compute / Hardware / Params | Source Citation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |

*(If multiple tasks/datasets exist, include a row for each key benchmark. If compute/hardware is omitted, write "Not reported".)*

### 2. Ablation Study Teardown (What Actually Drives the Gain?)
- Dissect the paper's ablation tables `[Cite Table/Sec.]`.
- Quantify the exact performance drop when each proposed component or loss term is removed.
- **Primary Driver Identification:** Explicitly identify which single component accounts for the majority of the performance improvement and which components yield marginal (<1%) gains.

### 3. Baseline Fairness & Evaluation Rigor Audit
- Evaluate whether baselines were tuned fairly (same parameter budget, training tokens, resolution, or inference compute) `[Cite Sec./Table]`.
- Point out any anomalies, missing standard baselines, or saturated benchmarks.
"""


PROMPT_BOTTLENECKS_AND_CRITIQUE = """Analyze the provided research paper context and produce **Module 4: Bottlenecks, Failure Modes & Architectural Critique**.

Think like a Principal Systems & ML Architect stress-testing this paper for production deployment. Structure your output using these exact Markdown sections:

### 1. Systems, Memory & Compute Bottlenecks
- **Computational Complexity:** Analyze time and space complexity (e.g., $\\mathcal{O}(N^2)$, FLOPs, matrix multiplications) during training and inference `[Cite Sec./Eq.]`.
- **Hardware & Memory Bandwidth Traps:** Identify potential GPU SRAM/HBM memory-bandwidth bottlenecks, KV-cache overhead, irregular sparse memory access patterns, synchronization barriers, or training instabilities.

### 2. Theoretical & Empirical Weaknesses
- **Fragile Assumptions:** Identify mathematical assumptions (e.g., i.i.d., convexity, stationary distributions, rank assumptions) that break in real-world data `[Cite Sec./Eq.]`.
- **Empirical Blindspots:** Highlight narrow dataset scope, lack of out-of-distribution (OOD) testing, sensitivity to unstated hyperparameters, or failure modes mentioned (or buried) in the text.

### 3. Reproducibility Audit
- List every missing implementation detail required to reproduce the paper from scratch (e.g., exact optimizer states, warmup steps, data filtering heuristics, random seeds, hardware footprint).
- Rate overall reproducibility (**High / Medium / Low**) with justification.

### 4. Three Concrete Engineering Fixes & Next-Gen Improvements
- Propose **3 actionable, deeply technical architectural or systems modifications** (e.g., kernel fusion, low-rank approximation, quantization-aware routing, alternative loss regularization) to resolve the bottlenecks identified above.
"""


PROMPT_CHAT_SYSTEM = """You are a Principal AI Research Assistant answering technical questions about a loaded research paper.
Use ONLY the provided paper sections and extracted tables to answer the user's question.
- Always cite exact sections, equations, and tables in brackets, e.g., `[Sec. 3.1, Eq. 2, Table 1]`.
- Use LaTeX (`$...$` and `$$...$$`) for all equations and mathematical notation.
- If the answer is not contained in the paper text, state clearly that the paper does not report that information before offering any theoretical derivation.
"""


def build_module_user_prompt(
    paper_title: str,
    task_instructions: str,
    context_text: str,
    tables_markdown: str = "",
    execution_note: str = "",
) -> str:
    """Constructs the structured user message containing paper context and tables."""
    parts = [
        f"# PAPER TITLE: {paper_title}",
    ]
    if execution_note:
        parts.append(f"*{execution_note}*")

    parts.append("## PAPER CONTENT (READING-ORDER MARKDOWN)\n" + context_text.strip())

    if tables_markdown and tables_markdown.strip():
        parts.append("## EXPLICITLY EXTRACTED BENCHMARK & ABLATION TABLES\n" + tables_markdown.strip())

    parts.append("---\n## YOUR ANALYSIS TASK\n" + task_instructions.strip())
    return "\n\n".join(parts)
