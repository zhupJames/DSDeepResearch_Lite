#!/usr/bin/env python3
import os
import json
import re
import time
import traceback
import threading, uuid
import requests
from flask import Flask, render_template, request, flash, jsonify

# --- Configuration and API Keys ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENROUTER_API_KEY:
    raise Exception("CRITICAL ERROR: OPENROUTER_API_KEY environment variable not set.")

DEEPSEEK_MODEL = "deepseek/deepseek-chat"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_ENDPOINT = "https://serpapi.com/search"

# --- Default Constants ---
DEFAULT_MAX_TOKENS_SYNTHESIS_PER_SECTION = 500
DEFAULT_MAX_TOKENS_OUTLINE = 1500
DEFAULT_MAX_TOKENS_QUERY_GENERATION = 300
DEFAULT_MAX_TOKENS_QUERY_REFINEMENT = 100

DEFAULT_TARGET_SOURCE_COUNT = 40
DEFAULT_SEARCH_RESULTS_PER_QUERY = 5
DEFAULT_MAX_SEARCH_ATTEMPTS = 10
DEFAULT_MAX_QUERY_REFINEMENT_ATTEMPTS = 3

# --- Language Strings ---
LANG_STRINGS = {
    "en": {
        "not_generated": "Not generated.",
        "section_not_generated": "\n*Content not generated for this section.*\n",
        "references_heading": "Collected Sources (via SerpApi)",
        "no_references": "[No sources collected or SerpApi key missing/invalid]",
    },
    "zh": {
        "not_generated": "未生成。",
        "section_not_generated": "\n*此部分内容未生成。*\n",
        "references_heading": "收集的来源 (通过 SerpApi)",
        "no_references": "[未收集到来源或SerpApi密钥丢失/无效]",
    }
}

# --- Core Functions ---

def call_llm_api(prompt, lang_code="en", model=DEEPSEEK_MODEL, max_tokens=2000, temperature=0.7):
    """
    Sends a prompt to the LLM API and returns the response.
    """
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured."
    system_message = f"Advanced research assistant for language {lang_code}."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "DSDeepResearch_Web",
        "X-Title": "DSDeepResearch_Web"
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and result["choices"]:
            content = result["choices"][0]["message"]["content"]
            return content.strip()
        return "Error: Unexpected API response format"
    except Exception as e:
        return f"Error: API request failed ({e})"

def perform_external_search(query: str, num_results: int = DEFAULT_SEARCH_RESULTS_PER_QUERY):
    """
    Performs a general web search using SerpApi.
    Returns a dictionary with 'text' and 'source_url' or None.
    """
    if not SERPAPI_API_KEY:
        print("No SERPAPI_API_KEY provided. Skipping web search.")
        return None
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
        "hl": "en",
        "engine": "google"
    }
    try:
        response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=60)
        response.raise_for_status()
        results = response.json()
        if "organic_results" in results and results["organic_results"]:
            first_result = results["organic_results"][0]
            snippet = first_result.get("snippet", "No snippet available.")
            url = first_result.get("link")
            title = first_result.get("title", "Unknown Source")
            if url:
                return {"text": f"Title: {title}\nSnippet: {snippet}", "source_url": url}
        return None
    except Exception as e:
        print(f"Search error: {e}")
        return None

def parse_outline_sections(outline_text):
    """
    Parses the outline text using a simple regex.
    Returns a list of section strings.
    """
    pattern = r"^\s*(?:[IVXLCDM]+\.[ ]+|[A-Z]\.[ ]+|[0-9]+\.[ ]+|[a-z]\.[ ]+|[一二三四五六七八九十]+[、．\.])\s+.*"
    sections = []
    if not isinstance(outline_text, str):
        return sections
    for line in outline_text.splitlines():
        if re.match(pattern, line.strip()):
            sections.append(line.strip())
    if not sections:
        sections = [line.strip() for line in outline_text.splitlines() if line.strip()]
    return sections

def run_research_process(topic, lang_code="en", target_model_id=DEEPSEEK_MODEL, 
                         target_sources=DEFAULT_TARGET_SOURCE_COUNT,
                         max_searches=DEFAULT_MAX_SEARCH_ATTEMPTS,
                         study_mode="testing",
                         max_refinements=DEFAULT_MAX_QUERY_REFINEMENT_ATTEMPTS,
                         max_tokens_outline=DEFAULT_MAX_TOKENS_OUTLINE,
                         max_tokens_synthesis=DEFAULT_MAX_TOKENS_SYNTHESIS_PER_SECTION,
                         search_results_per_q=DEFAULT_SEARCH_RESULTS_PER_QUERY,
                         max_tokens_query_gen=DEFAULT_MAX_TOKENS_QUERY_GENERATION,
                         max_tokens_query_refine=DEFAULT_MAX_TOKENS_QUERY_REFINEMENT):
    """
    Executes the research process:
      1. Generates search queries via the LLM.
      2. Gathers sources via SerpApi.
      3. Generates an outline.
      4. Synthesizes section content.
      5. Assembles the final document.
    
    Returns a dictionary containing all research data and the assembled document.
    """
    research_data = {
        "topic": topic,
        "sources": [],
        "outline": "",
        "synthesis": {},
        "estimated_char_count": 0
    }
    # --- Step 1: Generate Search Queries ---
    query_gen_prompt = f"""
Based on the research topic "{topic}", generate a list of 5 diverse and optimized English Google search queries.
Output ONLY the list of queries, one query per line.
    """
    llm_generated_queries_str = call_llm_api(query_gen_prompt, lang_code="en", model=target_model_id, max_tokens=max_tokens_query_gen)
    search_queries = []
    if llm_generated_queries_str and not llm_generated_queries_str.startswith("Error:"):
        search_queries = [q.strip() for q in llm_generated_queries_str.splitlines() if q.strip()]
        search_queries = [q for q in search_queries if len(q) > 3 and not q.startswith("Here") and not q.startswith("1.")]
    if not search_queries:
        search_queries = [f'"{topic}"', f'"{topic}" review', f'"{topic}" survey']

    gathered_sources = {}
    query_attempts = 0
    refinement_attempts = 0
    query_index = 0
    while len(gathered_sources) < target_sources and query_attempts < max_searches:
        current_query = search_queries[query_index % len(search_queries)]
        query_attempts += 1
        search_result = perform_external_search(current_query, num_results=search_results_per_q)
        if search_result and search_result.get("source_url"):
            source_url = search_result["source_url"]
            if source_url not in gathered_sources:
                gathered_sources[source_url] = search_result
        else:
            if refinement_attempts < max_refinements:
                refinement_attempts += 1
                refinement_prompt = f"""
The previous English Google search query '{current_query}' for topic '{topic}' failed to return useful results.
Please generate an alternative English search query that is substantially different.
Output only the new query string.
                """
                new_query_str = call_llm_api(refinement_prompt, lang_code="en", model=target_model_id, max_tokens=max_tokens_query_refine)
                if new_query_str and not new_query_str.startswith("Error:"):
                    search_queries[query_index % len(search_queries)] = new_query_str.strip('\'" \n\t')
        query_index += 1
        time.sleep(1.5)
    research_data["sources"] = list(gathered_sources.values())
    if not research_data["sources"]:
        raise Exception("Could not gather any sources.")

    # --- Step 2: Generate Outline ---
    source_summary = ""
    for i, src in enumerate(research_data["sources"]):
        source_summary += f"Source {i+1} ({src.get('source_url', 'N/A')}):\n{src.get('text', 'N/A')[:500]}...\n\n"
    outline_prompt = f"""
Based on the following {len(research_data['sources'])} collected sources, generate a logical outline in {lang_code.upper()}.
Collected Source Summaries:
---
{source_summary[:10000]}
---
    """
    research_data["outline"] = call_llm_api(outline_prompt, lang_code=lang_code, model=target_model_id, max_tokens=max_tokens_outline)
    if research_data["outline"].startswith("Error:"):
        raise Exception("Failed to generate outline from sources.")
    
    # --- Step 3: Section-by-Section Synthesis ---
    outline_sections = parse_outline_sections(research_data["outline"])
    sections_to_process = outline_sections[:2] if study_mode == "testing" else outline_sections
    for section in sections_to_process:
        relevant_info = ""
        for idx, src in enumerate(research_data["sources"]):
            relevant_info += f"Source {idx+1}:\n{src.get('text', 'N/A')[:1500]}\n(URL: {src.get('source_url', 'N/A')})\n\n"
        synthesis_prompt = f"""
Objective: Write a concise but informative section in {lang_code.upper()} on: "{section}".
Use the provided source material below:
---
{relevant_info[:10000]}
---
Please synthesize the content and include citations where appropriate.
        """
        synthesized_content = call_llm_api(synthesis_prompt, lang_code=lang_code, model=target_model_id, max_tokens=max_tokens_synthesis)
        if not synthesized_content.startswith("Error:"):
            research_data["synthesis"][section] = synthesized_content
            research_data["estimated_char_count"] += len(synthesized_content)
        else:
            research_data["synthesis"][section] = f"Error generating content for {section}."

    # --- Assemble Final Document ---
    lang_strings = LANG_STRINGS.get(lang_code, LANG_STRINGS["en"])
    final_content_warning = (
        "(Content based on gathered sources via SerpApi - General Search)"
        if SERPAPI_API_KEY and research_data.get("sources")
        else "(Web search skipped or failed; content may be limited)"
    )
    full_text = [f"# Research Document: {research_data['topic']}\n"]
    full_text.append(f"\n## Outline ({lang_code.upper()}):")
    full_text.append(research_data.get("outline", lang_strings["not_generated"]))
    full_text.append(f"\n## Synthesized Content {final_content_warning}")
    processed_sections = list(research_data["synthesis"].keys())
    ordered_sections = []
    for sec in parse_outline_sections(research_data["outline"]):
        if sec in processed_sections:
            ordered_sections.append(sec)
    for sec in processed_sections:
        if sec not in ordered_sections:
            ordered_sections.append(sec)
    if ordered_sections:
        for sec in ordered_sections:
            content = research_data["synthesis"].get(sec, lang_strings["section_not_generated"])
            full_text.append(f"\n### {sec}")
            full_text.append(content)
    else:
        full_text.append("\n[No sections were synthesized in this run.]")
    full_text.append(f"\n\n## {lang_strings['references_heading']}\n")
    if research_data.get("sources"):
        for i, src in enumerate(sorted(research_data["sources"], key=lambda x: x.get("source_url", ""))):
            full_text.append(f"{i+1}. URL: {src.get('source_url', 'N/A')}\n   Info: {src.get('text', '')[:120]}...")
    else:
        full_text.append(lang_strings["no_references"])
    assembled_document = "\n".join(full_text)
    research_data["assembled_document"] = assembled_document
    return research_data

# --- Background Job Management ---
job_store = {}

def async_research_process(job_id, params):
    try:
        research_data = run_research_process(**params)
        assembled_doc = research_data.get("assembled_document", "No document assembled.")
        job_store[job_id]["status"] = "complete"
        job_store[job_id]["result"] = assembled_doc
    except Exception as e:
        job_store[job_id]["status"] = "error"
        job_store[job_id]["result"] = f"An error occurred: {str(e)}\n{traceback.format_exc()}"

# --- Flask Application Setup ---
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Change to a strong secret for production use

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/start_research", methods=["POST"])
def start_research():
    topic = request.form.get("topic", "").strip()
    if not topic:
        flash("Please enter a research topic.", "error")
        return render_template("index.html")
    
    try:
        params = {
            "topic": topic,
            "lang_code": request.form.get("language", "en"),
            "study_mode": request.form.get("study_mode", "testing"),
            "target_sources": int(request.form.get("target_sources", DEFAULT_TARGET_SOURCE_COUNT)),
            "max_searches": int(request.form.get("max_searches", DEFAULT_MAX_SEARCH_ATTEMPTS)),
            "search_results_per_q": int(request.form.get("results_per_query", DEFAULT_SEARCH_RESULTS_PER_QUERY)),
            "max_refinements": int(request.form.get("max_refinements", DEFAULT_MAX_QUERY_REFINEMENT_ATTEMPTS)),
            "max_tokens_outline": int(request.form.get("max_tokens_outline", DEFAULT_MAX_TOKENS_OUTLINE)),
            "max_tokens_synthesis": int(request.form.get("max_tokens_synthesis", DEFAULT_MAX_TOKENS_SYNTHESIS_PER_SECTION)),
            "max_tokens_query_gen": int(request.form.get("max_tokens_query_gen", DEFAULT_MAX_TOKENS_QUERY_GENERATION)),
            "max_tokens_query_refine": int(request.form.get("max_tokens_query_refine", DEFAULT_MAX_TOKENS_QUERY_REFINEMENT)),
            "target_model_id": DEEPSEEK_MODEL
        }
    except Exception as e:
        flash("Error parsing parameters.", "error")
        return render_template("index.html")
    
    # Create a unique job ID and store initial status
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "processing", "result": None}
    # Start background thread for the research process
    thread = threading.Thread(target=async_research_process, args=(job_id, params), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})

@app.route("/job_status/<job_id>")
def job_status(job_id):
    job = job_store.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)

@app.route("/results/<job_id>")
def results(job_id):
    job = job_store.get(job_id)
    if not job or job["status"] != "complete":
        flash("Job not complete or not found.", "error")
        return render_template("index.html")
    assembled_doc = job.get("result", "No result available.")
    return render_template("results.html", assembled_doc=assembled_doc)

if __name__ == "__main__":
    app.run(debug=True)
