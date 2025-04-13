# DSDeepResearch_Lite

## Description

This Python application provides a graphical user interface (GUI) built with Tkinter to automate parts of the research process. It leverages Large Language Models (LLMs) via OpenRouter (specifically configured for DeepSeek models) and web search via the SerpApi service to:

1.  Generate optimized search queries based on a user-provided topic.
2.  Gather source information (titles, snippets, URLs) from the web using SerpApi (limited attempts).
3.  Attempt to refine search queries using the LLM if initial searches fail.
4.  Generate a hierarchical outline based on the gathered source information using the LLM.
5.  Synthesize content for the initial sections of the outline based on the gathered sources, using the LLM.
6.  Support output generation in English or Chinese (Simplified).
7.  Save the collected data (JSON) and the assembled report (TXT) to a specified directory.

This script follows a **source-first** workflow, attempting to gather information before generating the outline and content.

## Features

* Simple GUI for topic input and language selection (English/Chinese).
* Uses LLM (via OpenRouter) to generate optimized search queries.
* Performs general web search using SerpApi (requires API key).
* Attempts LLM-based query refinement upon search failure (limited).
* Generates an outline based on collected source snippets.
* Synthesizes content based on sources, with basic inline citation support structure.
* Outputs progress and results to the GUI window.
* Saves structured data (`.json`) and the final report (`.txt`) to a configurable directory.
* Includes placeholders for future PDF text extraction.

## Prerequisites

* **Python 3.x:** Ensure you have a recent version of Python 3 installed.
* **`requests` Library:** This script uses the `requests` library to make API calls.
* **OpenRouter API Key:** You need an API key from [OpenRouter.ai](https://openrouter.ai/) to use the LLM for query generation, outlining, and synthesis.
* **SerpApi API Key:** You need an API key from [SerpApi.com](https://serpapi.com/) to perform the web searches in Step 1. The script will run without it, but the source gathering step will be skipped.

## Setup

1.  **Save the Script:** Download or copy the Python code and save it as a file (e.g., `research_gui.py`).
2.  **Install Dependencies:** Open your terminal or command prompt and install the `requests` library:
    ```bash
    pip install requests
    ```
    *(Tkinter is typically included with Python)*
3.  **Set Environment Variables:** This is the most crucial step for API keys. You **must** set the following environment variables before running the script:
    * `OPENROUTER_API_KEY`: Set this to your OpenRouter API key.
    * `SERPAPI_API_KEY`: Set this to your SerpApi API key.

    *How to set environment variables depends on your OS:*
    * **Windows:** Search for "Environment Variables" in the Start Menu -> "Edit the system environment variables" -> "Environment Variables..." button -> Add/Edit under "User variables". **Remember to close and reopen any terminal/IDE after setting variables.**
    * **Linux/macOS (Temporary for current session):**
        ```bash
        export OPENROUTER_API_KEY='your_openrouter_key'
        export SERPAPI_API_KEY='your_serpapi_key'
        ```
    * **Linux/macOS (Permanent):** Add the `export` lines above to your shell profile file (e.g., `~/.bashrc`, `~/.zshrc`, `~/.profile`) and restart your terminal or run `source ~/.bashrc` (or equivalent).

## Usage

1.  **Run the Script:** Open a terminal or command prompt (a *new* one after setting environment variables) navigate to the directory where you saved the file, and run:
    ```bash
    python research_gui.py
    ```
    *(Replace `research_gui.py` with your filename if different)*
2.  **Enter Topic:** Type your research topic into the "Research Topic" field in the GUI window that appears.
3.  **Select Language:** Choose either "English" or "Chinese (简体)" using the radio buttons.
4.  **Start Research:** Click the "Start Research" button. The button will disable while processing.
5.  **Monitor Output:** Watch the text area below for progress updates, logs, potential errors, and the final assembled report draft.
6.  **Find Files:** Once finished, the script will save two files (`research_data_...json` and `research_report_...txt`) into the directory specified by the `target_directory` variable in the script (see Configuration section).

## Configuration (Inside the Script)

You can modify the following constants near the top of the script file:

* `DEEPSEEK_MODEL`: Change the specific LLM model ID used via OpenRouter.
* `OPENROUTER_API_URL`, `SERPAPI_ENDPOINT`: API endpoint URLs (usually don't need changing).
* `LLM_HEADERS`: Update `HTTP-Referer` and `X-Title` if desired for OpenRouter.
* `MAX_TOKENS_*`: Adjust token limits for different API calls (lower values = faster/cheaper but less detailed output).
* `TARGET_SOURCE_COUNT`: The desired number of sources to find (may not be reached).
* `SEARCH_RESULTS_PER_QUERY`: How many results SerpApi returns per query.
* `MAX_SEARCH_ATTEMPTS`: Maximum number of SerpApi calls allowed per run.
* `MAX_QUERY_REFINEMENT_ATTEMPTS`: Maximum number of times the LLM will be asked to refine a failed query.
* `target_directory` (in the `if __name__ == "__main__":` block): Change the path where output files are saved. **Ensure this path exists or the script has permissions to create it.**

## Limitations & Known Issues

* **Web Search Dependency:** Relies entirely on a valid `SERPAPI_API_KEY`. If not provided or invalid, source gathering fails.
* **PDF Extraction Placeholder:** The `extract_text_from_pdf` function is a placeholder. Implementing it requires adding PDF parsing libraries (`PyPDF2`, `pdfminer.six`) and handling potential download/parsing errors. Currently, only search snippets are used.
* **Outline Parsing:** The `parse_outline_sections` function uses basic regular expressions and may not correctly parse all outline formats generated by the LLM.
* **Search Quality:** The effectiveness of the source gathering depends heavily on the quality of the search queries generated by the LLM and the results returned by SerpApi. Finding 40 relevant sources is not guaranteed.
* **Basic Error Handling:** Error handling for API calls and file operations is basic. More robust handling could be added.
* **GUI Responsiveness:** While threading is used, the GUI might still become slightly unresponsive during intensive processing or if file saving takes time (saving is currently in the main thread after the worker finishes).

## Potential Future Improvements

* Implement PDF downloading and text extraction in `extract_text_from_pdf`.
* Implement more robust outline parsing.
* Add options to configure search parameters (e.g., date range, specific sites beyond arXiv) via the GUI.
* Improve error handling and user feedback in the GUI.
* Integrate embedding generation and retrieval (RAG) for more context-aware synthesis.
* Allow saving/loading of research sessions.
* Refactor code into more modular classes/functions.

## License

Copyright 2025 Pengwei Zhu Permission is hereby granted, free of charge, to any person obtaining a copyof this software and associated documentation files (the "Software"), to dealin the Software without restriction, including without limitation the rightsto use, copy, modify, merge, publish, distribute, sublicense, and/or sellcopies of the Software, and to permit persons to whom the Software isfurnished to do so, subject to the following conditions:The above copyright notice and this permission notice shall be included in allcopies or substantial portions of the Software.THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS ORIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
