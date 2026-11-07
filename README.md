#  Paper Scalper : A Research Paper Deep Technical Analyzer

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit%201.42-FF4B4B.svg)
![PyMuPDF4LLM](https://img.shields.io/badge/PDF-PyMuPDF4LLM-success.svg)
![Local AI](https://img.shields.io/badge/Local%20AI-Ollama%20%7C%20llama.cpp-orange.svg)
![Cloud APIs](https://img.shields.io/badge/Cloud-Kimi%20%7C%20OpenAI%20%7C%20Gemini%20%7C%20Claude-purple.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A production-grade **Research Paper Analyzer & Technical Teardown Studio** built for ML engineers, systems researchers, and peer reviewers. 

Standard PDF text extractors scramble two-column academic layouts, mangle multi-line LaTeX equations, and flatten benchmark tables into unreadable text streams. Furthermore, feeding an entire 15–30 page paper (plus 4 pages of references) into a local 7B/8B LLM frequently triggers context truncation or VRAM out-of-memory (OOM) errors.

This application solves both problems through a **layout-aware PyMuPDF ingestion pipeline** paired with a **Hybrid Local/Cloud LLM Router** that dynamically switches between **Full-Context Analysis** and **VRAM-Safe Section-Routed Map-Reduce**.

---

##  Key Architectural Features

### 1. Layout-Aware Academic PDF & Table Parser (`parser.py`)
- **Two-Column Reading Order:** Uses `pymupdf4llm` and `fitz` (PyMuPDF) to reconstruct natural reading order across complex multi-column conference templates (NeurIPS, ICML, ICLR, CVPR, ACL, IEEE, arXiv).
- **Dual-Pass Benchmark Table Extractor:** Combines PyMuPDF's native geometric table finder (`page.find_tables()`) with Markdown pipe-table regex extraction so ablation and SOTA comparison numbers are isolated intact.
- **Canonical Section Segmentation:** Automatically detects and buckets the paper into `Abstract`, `Introduction`, `Related Work`, `Methodology`, `Experiments & Benchmarks`, `Limitations`, `Conclusion`, and `Appendix`.
- **Automatic Reference Stripper:** Isolates and strips the `References` / `Bibliography` section prior to LLM ingestion, reducing prompt token footprint and KV-cache VRAM consumption by **20–30%**.

### 2. Hybrid Local + Cloud Inference Engine (`llm_engine.py`)
- **Local AI Mode (100% On-Device):** Native support for **Ollama** (`http://localhost:11434/v1`) and **llama.cpp `llama-server`** (`http://localhost:8080/v1`) with hardware-aware `num_ctx` injection and DeepSeek-R1 `reasoning_content` streaming support.
- **Cloud API Mode:** Seamless integration with **Moonshot / Kimi** (`https://api.moonshot.ai/v1`), **OpenAI**, **Google Gemini**, **Anthropic Claude**, or any custom OpenAI-compatible endpoint.
- **Live Endpoint Discovery:** One-click model detection queries `/v1/models` on your local server or cloud provider to populate available model IDs automatically.

### 3. Smart Execution Strategy (`analyzer.py`)
- **Full-Context Pass:** Sends the complete cleaned Markdown paper + all extracted benchmark tables in a single pass when running on large-context Cloud APIs (128k–200k context).
- **Section-Routed Map-Reduce:** Automatically routes only the highest-signal sections to each specialized analysis module when running on local 7B/8B models or constrained context windows (8k–16k), preventing mid-paper truncation and VRAM spikes.

| Analysis Module | Routed Paper Sections (Map-Reduce Mode) | Extracted Tables Attached? |
| :--- | :--- | :---: |
| **1. Executive Summary & Core Delta** | `Abstract` + `Introduction` + `Conclusion` | Yes |
| **2. Topics, Taxonomy & Math Formulation** | `Abstract` + `Methodology` + `Introduction` + `Appendix` | No |
| **3. Benchmarks & Ablations Teardown** | `Experiments & Benchmarks` + `Abstract` + `Conclusion` | Yes |
| **4. Bottlenecks & Architectural Critique** | `Methodology` + `Experiments & Benchmarks` + `Limitations` + `Conclusion` | Yes |
| **5. Grounded Paper Chat (Q&A)** | Top keyword-scored sections + `Abstract` | Yes |

### 4. Strict Anti-Hallucination Prompts (`prompts.py`)
- Forces inline bracketed citations for every claim, equation, and metric (`[Sec. 3.2, Eq. 4, Table 2]`).
- Explicitly flags omitted hyperparameter, hardware, or complexity details as **"Not reported in paper"** rather than guessing.
- Generates a structured **Reproducibility Audit** and **3 Concrete Engineering Fixes** for identified systems/compute bottlenecks.

---

##  System Architecture

```text
                       ┌────────────────────────────┐
                       │   Uploaded Academic PDF    │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
               ┌──────────────────────────────────────────┐
               │       AcademicPaperParser (parser.py)    │
               ├──────────────────────────────────────────┤
               │ • pymupdf4llm 2-column reading order     │
               │ • Geometric + Markdown table extraction  │
               │ • Canonical section segmentation         │
               │ • Reference/Bibliography token stripper  │
               └──────┬────────────────────────────┬──────┘
                      │                            │
         Cleaned Sections Dict            Extracted Tables Digest
                      │                            │
                      └──────────────┬─────────────┘
                                     ▼
               ┌──────────────────────────────────────────┐
               │         PaperAnalyzer (analyzer.py)      │
               ├──────────────────────────────────────────┤
               │ Strategy Resolver:                       │
               │  ├─ Full-Context Pass (Cloud / >24k ctx) │
               │  └─ Section-Routed Map-Reduce (Local AI) │
               └─────────────────────┬────────────────────┘
                                     │
                                     ▼
               ┌──────────────────────────────────────────┐
               │        HybridLLMEngine (llm_engine.py)   │
               ├─────────────────────┬────────────────────┤
               │      Local AI       │     Cloud APIs     │
               │ • Ollama (:11434)   │ • Moonshot / Kimi  │
               │ • llama.cpp (:8080) │ • OpenAI / Gemini  │
               └─────────────────────┴────────────────────┘
                                     │
                                     ▼
               ┌──────────────────────────────────────────┐
               │      Streamlit Workspace UI (app.py)     │
               ├──────────────────────────────────────────┤
               │ • 5 Live-Streaming Analysis & Chat Tabs  │
               │ • One-Click .md & .json Report Exports   │
               └──────────────────────────────────────────┘
```

---

## Project Structure

```text
research-paper-analyzer/
├── requirements.txt      # Pinned production dependencies
├── .env.example          # Optional environment variable template
├── README.md             # Project documentation
├── parser.py             # Layout-aware PDF, section, equation & table parser
├── prompts.py            # Anti-hallucination system prompts & task templates
├── llm_engine.py         # Unified Local (Ollama/llama.cpp) & Cloud OpenAI SDK router
├── analyzer.py           # Full-Context vs. Section-Routed Map-Reduce orchestrator
└── app.py                # Streamlit interactive UI with live token streaming & export
```

---

## Installation & Setup

### 1. Clone & Create a Virtual Environment

```bash
git clone [https://github.com/your-username/research-paper-analyzer.git](https://github.com/your-username/research-paper-analyzer.git)
cd research-paper-analyzer

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional)

Create a `.env` file in the root directory if you want pre-filled Cloud API keys or custom defaults:

```ini
# Optional Cloud API Keys
MOONSHOT_API_KEY="sk-your-moonshot-api-key"
OPENAI_API_KEY="sk-your-openai-api-key"
GEMINI_API_KEY="your-gemini-api-key"

# Optional Custom Defaults
LLM_BASE_URL="http://localhost:11434/v1"
LLM_MODEL_NAME="qwen2.5:7b-instruct"
```

---

##  Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The web UI will open automatically at `http://localhost:8501`.

---

##  Backend Configuration Guides

### Option A: Local AI with Ollama
1. Start the Ollama daemon and pull an instruction or reasoning model:
   ```bash
   ollama serve
   ollama pull qwen2.5:7b-instruct
   # Other strong candidates:
   # ollama pull llama3.1:8b
   # ollama pull deepseek-r1:8b
   ```
2. In the app sidebar:
   - Select **Local AI (Ollama / llama.cpp)** $\rightarrow$ **Local AI — Ollama (`http://localhost:11434/v1`)**.
   - Click ** Detect Models** and select your pulled model.
   - Keep **Execution Mode** set to `Auto (Smart Context Routing)` and **Strip References to Save VRAM** enabled.

### Option B: Local AI with `llama.cpp` (`llama-server`)
1. Launch `llama-server` with an OpenAI-compatible endpoint and an 8k or 16k context window:
   ```bash
   ./llama-server \
     -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
     --host 127.0.0.1 \
     --port 8080 \
     -c 16384 \
     --ngl 99
   ```
2. In the app sidebar:
   - Select **Local AI (Ollama / llama.cpp)** $\rightarrow$ **Local AI — llama.cpp Server (`http://localhost:8080/v1`)**.
   - Match the sidebar **Context Window (`num_ctx`)** slider to the `-c` flag passed to `llama-server`.

### Option C: Cloud Frontier APIs (Kimi / Moonshot, OpenAI, Gemini, Claude)
1. In the app sidebar, switch **Inference Backend** to **Cloud API**.
2. Choose your preset:
   - **Moonshot / Kimi:** `https://api.moonshot.ai/v1` (e.g., `kimi-k2-0711-preview` or `moonshot-v1-128k`)
   - **OpenAI:** `https://api.openai.com/v1` (e.g., `gpt-4o`, `o3-mini`)
   - **Google Gemini:** `https://generativelanguage.googleapis.com/v1beta/openai/` (e.g., `gemini-2.5-pro`)
   - **Anthropic Claude:** `https://api.anthropic.com/v1/` (e.g., `claude-3-7-sonnet-latest`)
3. Enter your API key (if not already loaded via `.env`).
4. Set **Execution Mode** to `Auto` or `Full-Context Pass`.

---

## Hardware & VRAM Tuning Guide (Local Models)

| GPU VRAM | Recommended Local Quantization | Recommended `num_ctx` | Execution Mode | Strip References? |
| :--- | :--- | :---: | :--- | :---: |
| **3 GB – 4 GB** | `Qwen2.5-3B-Instruct (Q4_K_M)` | `8192` | `Section-Routed Map-Reduce` | ✅ Required |
| **6 GB – 8 GB** | `Qwen2.5-7B-Instruct (Q4_K_M)` / `Llama-3.1-8B (Q4_K_M)` | `16384` | `Section-Routed Map-Reduce` | ✅ Recommended |
| **12 GB – 16 GB** | `Qwen2.5-14B-Instruct (Q4_K_M)` / `DeepSeek-R1-14B` | `16384 – 32768` | `Auto` | ✅ Recommended |
| **24 GB+** | `Qwen2.5-32B-Instruct (Q4_K_M)` | `32768 – 65536` | `Full-Context Pass` | Optional |

---

##  Exporting Reports

Once one or more analysis modules finish streaming:
- Click **Export Report (`.md`)** to download a single formatted Markdown file containing paper telemetry, all 4 teardown modules, and an appendix of extracted benchmark tables.
- Click **Export Data (`.json`)** to download a structured JSON document containing token statistics, detected section names, raw table schemas, and module outputs for downstream pipelines or vector indexing.

---

##  License

Released under the **MIT License**. Free for academic, personal, and commercial use.
