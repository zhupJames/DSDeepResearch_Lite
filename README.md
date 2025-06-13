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

## Potential Future Improvements

- PDF text extraction and richer source processing.
- More robust outline parsing.
- Additional search configuration options.
- Enhanced error handling across interfaces.
- Ability to save and resume research sessions.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
