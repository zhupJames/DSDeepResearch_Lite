# DSDeepResearch_Lite

DSDeepResearch_Lite is a minimal research assistant designed around the
**DeepSeek** large language model.  The codebase demonstrates how to send
prompts to DeepSeek via the OpenRouter API, gather sources with SerpApi and then
generate a short research report.  Several interfaces are provided so the same
workflow can be used from the command line, in a GUI, as a small web service or
through a Model Context Protocol (MCP) endpoint.

The project is intentionally lightweight to make it easy to run locally.  Only a
handful of third‑party packages are required and most functionality resides in a
single script per interface.

## Project Layout

```
DS_Deepresearch.py          - Command line / IDE interface
DS_Deepresearch_GUI/        - Tkinter based desktop GUI
DS_Deepresearch_web/        - Flask web application
DS_Deepresearch_MCP/        - FastMCP server exposing the workflow
requirements.txt            - Python package requirements
tests/                      - Unit tests
```

## Prerequisites

* **Python 3.8+** (Python 3.9 or newer recommended)
* Internet connectivity for the OpenRouter and SerpApi endpoints
* API keys for:
  * **OpenRouter** – required to access the DeepSeek model
  * **SerpApi** – optional but enables web search for source collection

The GUI version requires Tkinter which is bundled with most Python installers.
The web and MCP variants use Flask and FastMCP respectively; both are installed
from `requirements.txt`.

### Installing

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your API keys either as environment variables or in a `.env` file placed
   in the project root:

   ```bash
   OPENROUTER_API_KEY=your_openrouter_key
   SERPAPI_API_KEY=your_serpapi_key   # optional for web search
   ```

## Interfaces

All versions share the same underlying workflow defined in
`run_research_process`.  The process uses **DeepSeek** for prompt generation and
summary writing and optionally SerpApi for gathering source material.  Content is
assembled into a short document saved in the `Deepresearch_Output` directory.

### Command Line

`DS_Deepresearch.py` provides an interactive CLI.  It prompts for a topic and
various parameters then executes the research workflow, saving a draft report in
JSON and text formats.

### Desktop GUI

`DS_Deepresearch_GUI/DS_Deepresearch_GUI.py` implements a Tkinter interface.
Users can configure parameters, view log output and monitor progress in a simple
window.  The core functions are identical to the CLI but run in a worker thread
so the interface remains responsive.

### Flask Web Application

`DS_Deepresearch_web/app.py` exposes the workflow as a small Flask service.  It
supports asynchronous jobs allowing the browser to poll for status updates.  Run
the app and open `http://127.0.0.1:5000` to start a research task in your
browser.

### MCP Server

`DS_Deepresearch_MCP/DS_Deepresearch_MCP.py` wraps the workflow with FastMCP.
This enables integration with other tools that speak the Model Context Protocol
for scripted or programmatic use.

## Core Functions

The major scripts share a number of helper functions:

* `call_llm_api` – sends a prompt to the DeepSeek model via OpenRouter.
* `perform_external_search` – queries SerpApi and returns snippets and URLs.
* `extract_text_from_pdf` – placeholder demonstrating where PDF handling would
  occur.
* `parse_outline_sections` – parses the outline returned by the LLM into
  individual section headers.
* `run_research_process` – orchestrates query generation, searching and content
  synthesis.  This is the heart of the project.

Additional functions appear in the MCP and web variants (`generate_outline`,
`synthesize_section`, `refine_search_query` and others) but they ultimately call
into the same workflow.

## Running Tests

Basic unit tests are provided for the outline parser.  Run all tests with:

```bash
python -m unittest discover -v
```

## License

This project is licensed under the [MIT License](LICENSE).
