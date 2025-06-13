# DSDeepResearch_Lite

DSDeepResearch_Lite is an experimental research assistant that uses large language models hosted on OpenRouter to build academic-style reports. The project includes several front ends (CLI, GUI, Web and MCP) which share the same core workflow for generating search queries, collecting sources with SerpApi and synthesizing the final document.

## Project Structure

```
DS_Deepresearch.py          - Command line / IDE interface
DS_Deepresearch_GUI/        - Tkinter based desktop GUI
DS_Deepresearch_web/        - Flask web application
DS_Deepresearch_MCP/        - FastMCP server exposing the workflow
requirements.txt            - Python package requirements
```

## Installation

1. **Python 3.8+** is required.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Tkinter is usually bundled with Python. The web and MCP versions require Flask and FastMCP which are listed in the requirements file.

### Environment variables

Set your API keys either in a `.env` file or as system environment variables:

```
OPENROUTER_API_KEY=your_openrouter_key
SERPAPI_API_KEY=your_serpapi_key   # optional for web search
```

## Usage

### CLI
Run the interactive script inside a terminal:
```bash
python DS_Deepresearch.py
```

### GUI
Launch the desktop interface:
```bash
python DS_Deepresearch_GUI/DS_Deepresearch_GUI.py
```

### Web App
Start the Flask application and open `http://127.0.0.1:5000` in your browser:
```bash
python DS_Deepresearch_web/app.py
```

### MCP Server
Expose the workflow through the Model Context Protocol:
```bash
python DS_Deepresearch_MCP/DS_Deepresearch_MCP.py
```

## Testing

Run the unit tests with:
```bash
python -m unittest discover -v
```

## License

This project is licensed under the [MIT License](LICENSE).
