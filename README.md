# DSDeepResearch_Lite

DSDeepResearch_Lite is an experimental research assistant that uses OpenRouter hosted language models and optional web search to generate outlines and draft academic style content. The repository contains several ways to run the tool depending on your preferred interface.

## Project Structure

```
DS_Deepresearch.py          - Interactive CLI/IDE version
DS_Deepresearch_GUI/        - Tkinter based desktop GUI
DS_Deepresearch_web/        - Flask web application
DS_Deepresearch_MCP/        - FastMCP server exposing the workflow as tools
requirements.txt            - Python package requirements
```

All versions share the same core workflow:
1. Generate search queries with the LLM.
2. Collect source snippets with SerpApi (if the API key is provided).
3. Produce a structured outline based on the gathered sources.
4. Synthesize section content using the outline and sources.
5. Save the gathered data and assembled report to disk.

## Prerequisites

- **Python 3.8+**
- An **OpenRouter** API key for access to LLM models.
- (Optional) a **SerpApi** API key for web search.
- Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```
  The GUI relies on Tkinter which is included with most Python distributions (on Linux you may need the `python3-tk` package). The Web and MCP versions require additional dependencies that are listed in `requirements.txt`.

Create a `.env` file or set environment variables so the scripts can read your API keys:

```
OPENROUTER_API_KEY=your_openrouter_key
SERPAPI_API_KEY=your_serpapi_key
```

## Tool Overview

### DSDeepResearch IDE (CLI)
File: `DS_Deepresearch.py`

Interactive command line program intended to be run inside a terminal or IDE. The script prompts for all parameters such as language, topic, number of sources, and token limits.

**Usage**
```bash
python DS_Deepresearch.py
```
The script walks you through the options and saves the output under the `Deepresearch_Output` directory by default.

### DSDeepResearch GUI
File: `DS_Deepresearch_GUI/DS_Deepresearch_GUI.py`

Desktop application built with Tkinter. It presents form fields for the topic, language selection, and advanced options.

**Usage**
```bash
python DS_Deepresearch_GUI/DS_Deepresearch_GUI.py
```
A window appears where you can configure the run and monitor progress in real time. Results are written to the folder configured in the interface.

### DSDeepResearch Web
Files under `DS_Deepresearch_web/`

Lightweight Flask application that runs the research process in a background thread. Open the web page, submit a topic, and check the results when the job completes.

**Usage**
```bash
python DS_Deepresearch_web/app.py
```
Then open `http://127.0.0.1:5000` in a browser. Enter the topic, choose the language, and start a job. The assembled document is displayed on completion.

### DSDeepResearch MCP
File: `DS_Deepresearch_MCP/DS_Deepresearch_MCP.py`

Exposes the research workflow through the Model Context Protocol using the `fastmcp` package. This version is meant for integration with other tools that speak MCP.

**Usage**
```bash
python DS_Deepresearch_MCP/DS_Deepresearch_MCP.py
```
The server starts in STDIO mode and waits for MCP requests. Consult the `fastmcp` documentation for client integration details.
=======
* **Python 3.x:** Ensure you have a recent version of Python 3 installed.
* **Required Libraries:**
    * `requests` (for making API calls)
    * `python-dotenv` (for loading API keys from a `.env` file)
* **OpenRouter API Key:** You need an API key from [OpenRouter.ai](https://openrouter.ai/) to use the LLM.
* **SerpApi API Key:** You need an API key from [SerpApi.com](https://serpapi.com/) to perform web searches. The script will warn and skip searches if this key is missing.

## Setup

1.  **Save the Script:** Download or copy DS_Deepresearch.py and save it as a file.
2.  **Create `requirements.txt`:** Create a file named `requirements.txt` in the same directory as the script and add the following lines:
    ```txt
    requests>=2.31.0
    python-dotenv>=1.0.0
    ```
3.  **Install Dependencies:** Open your terminal or command prompt, navigate to the project directory, and run:
    ```bash
    pip install -r requirements.txt
    ```
    *(Tkinter is typically included with Python)*
4.  **Set API Keys:** You **must** provide your API keys using **one** of the following methods:

    * **Method A: `.env` File (Recommended)**
        1.  Create a file named exactly `.env` in the *same directory* as DS_Deepresearch.py.
        2.  Add your API keys to the `.env` file, one per line, like this (replace `your_..._key` with your actual keys):
            ```dotenv
            OPENROUTER_API_KEY='your_openrouter_key'
            SERPAPI_API_KEY='your_serpapi_key'
            ```
        3.  **Important:** Add `.env` to your `.gitignore` file to prevent accidentally committing your secret keys to version control. The script (assuming it's updated to use `load_dotenv()`) will automatically load these keys when run.

    * **Method B: System Environment Variables**
        1.  Set the `OPENROUTER_API_KEY` and `SERPAPI_API_KEY` environment variables directly in your operating system.
        2.  *How to set environment variables depends on your OS:*
            * **Windows:** Search for "Environment Variables" in the Start Menu -> "Edit the system environment variables" -> "Environment Variables..." button -> Add/Edit under "User variables". **Remember to close and reopen any terminal/IDE after setting variables.**
            * **Linux/macOS (Temporary for current session):**
                ```bash
                export OPENROUTER_API_KEY='your_openrouter_key'
                export SERPAPI_API_KEY='your_serpapi_key'
                ```
            * **Linux/macOS (Permanent):** Add the `export` lines above to your shell profile file (e.g., `~/.bashrc`, `~/.zshrc`, `~/.profile`) and restart your terminal or run `source ~/.bashrc` (or equivalent).

## Usage

1.  **Run the Script:** Open a terminal or command prompt (a *new* one after setting environment variables or creating `.env`), navigate to the project directory, and run:
    ```bash
    python DS_Deepresearch_GUI/DS_Deepresearch_GUI.py
    ```
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
* **`.env` Loading:** Assumes the Python script is updated to import and call `load_dotenv()` from the `dotenv` library near the beginning to actually load variables from the `.env` file.

## Potential Future Improvements

- PDF text extraction and richer source processing.
- More robust outline parsing.
- Additional search configuration options.
- Enhanced error handling across interfaces.
- Ability to save and resume research sessions.

## License
Copyright (c) 2025 Pengwei Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
