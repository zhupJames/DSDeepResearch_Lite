#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Revised for Model Context Protocol (MCP) Server with Local Mode Support using FastMCP

import os
import sys
import requests
import json
from typing import Optional, Dict, Any

# Import the FastMCP server class
from fastmcp import FastMCP

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
DEEPSEEK_MODEL = "deepseek/deepseek-chat"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_ENDPOINT = "https://serpapi.com/search"

# Headers for OpenRouter LLM calls
LLM_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "YOUR_APP_URL_OR_NAME", # Optional: Replace
    "X-Title": "DSDeepResearch_Lite CLI" # Optional: Replace
}

DEFAULT_SEARCH_RESULTS_PER_QUERY = 5

# Initialize FastMCP server
mcp = FastMCP(name="DSDeepResearchMCP")

# --- Health Check Endpoint ---
@mcp.resource("health://status")
def health() -> Dict[str, str]:
    """
    Health check endpoint returning server status.
    """
    return {"status": "ok"}

# --- MCP Tool: call_llm_api (matches DS_Deepresearch.py) ---
@mcp.tool(name="call_llm_api")
def call_llm_api(
    prompt: str,
    lang_code: str = 'en',
    model: str = DEEPSEEK_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.7
) -> str:
    """
    Sends a prompt to the configured LLM API and returns the response content.
    This function matches the logic and signature of DS_Deepresearch.py:call_llm_api.
    All errors are allowed to surface naturally.
    """
    # Check for required API key
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured."
    # Compose the system message for the LLM
    system_message = f"You are an advanced research assistant responding in {lang_code}. Your primary goals are: 1. Accuracy & Detail 2. Instruction Adherence (including language: {lang_code}) 3. Consistency 4. Clarity 5. Conditional Citation (follow instructions in user prompt)"
    # Prepare the payload for the OpenRouter API
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    # Print diagnostic information for transparency
    print(f"\n--- Sending Prompt to LLM {model} via OpenRouter (Lang: {lang_code}) ---")
    print(f"Prompt (first 300 chars): {prompt[:300]}...")
    print(f"(Requesting max_tokens: {max_tokens})")
    # Make the POST request to the LLM API
    response = requests.post(
        OPENROUTER_API_URL,
        headers={**LLM_HEADERS, "Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json=payload,
        timeout=180
    )
    print(f"--- requests.post call returned (Status: {response.status_code}) ---")
    if response.status_code != 200:
        print(f"Error Response Text: {response.text}")
        response.raise_for_status()
    result = response.json()
    # Parse the response and return the content
    if "choices" in result and result["choices"] and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
        content = result["choices"][0]["message"]["content"]
        if "usage" in result:
            print(f"--- API Usage: {result['usage']} ---")
        return content.strip()
    else:
        print("Warning: API response status 200 but did not contain expected content path.")
        print("Full response:", json.dumps(result, indent=2))
        if "error" in result:
            error_message = result['error'].get('message', 'No message provided')
            try:
                if '\\u' in error_message:
                    error_message = error_message.encode('utf-8').decode('unicode_escape')
            except Exception:
                pass
            print(f"API Error Message: {error_message}")
            return f"Error: API returned an error - {result['error'].get('code', 'Unknown code')}"
        return "Error: Unexpected API response format"

# --- MCP Tool: perform_external_search (matches DS_Deepresearch.py) ---
@mcp.tool(name="perform_external_search")
def perform_external_search(
    query: str,
    num_results: int = DEFAULT_SEARCH_RESULTS_PER_QUERY
) -> Optional[Dict[str, Any]]:
    """
    Performs a general web search using the SerpApi Google Search API.
    Uses hl=en for searching. Returns {'text': ..., 'source_url': ...} or None.
    This function matches the logic and signature of DS_Deepresearch.py:perform_external_search.
    All errors are allowed to surface naturally.
    """
    if not SERPAPI_API_KEY:
        print("--- SerpApi API Key not provided. Skipping web search. ---")
        return None
    hl_param = 'en'
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
        "hl": hl_param,
        "engine": "google"
    }
    print(f"--- Calling SerpApi Search ---")
    print(f"--- Query: {query} ---")
    print(f"--- Num Results: {num_results}, Lang (hl): {hl_param} ---")
    response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    results = response.json()
    if "organic_results" in results and results["organic_results"]:
        first_result = results["organic_results"][0]
        snippet = first_result.get('snippet', 'No snippet available.')
        url = first_result.get('link')
        title = first_result.get('title', 'Unknown Source')
        if url:
            print(f"--- SerpApi Search Success. Top result: {url} ---")
            return {'text': f"Title: {title}\nSnippet: {snippet}", 'source_url': url}
        else:
            print("--- SerpApi Search Warning: Top result missing URL. ---")
            return {'text': f"Title: {title}\nSnippet: {snippet}", 'source_url': None}
    else:
        print("--- SerpApi Search returned no 'organic_results'. ---")
        if "error" in results:
            print(f"--- SerpApi returned an error message: {results['error']} ---")
        return None

# --- MCP Tool: generate_outline ---
@mcp.tool(name="generate_outline")
def generate_outline(
    topic: str,
    sources: list,
    lang_code: str = 'en',
    max_tokens: int = 1500,
    model: str = DEEPSEEK_MODEL
) -> str:
    """
    Generates a logical outline for an academic paper based on the topic and provided sources.
    Args:
        topic (str): The research topic.
        sources (list): List of source dicts with 'text' and 'source_url'.
        lang_code (str): Output language code.
        max_tokens (int): Max tokens for LLM call.
        model (str): LLM model ID.
    Returns:
        str: The generated outline.
    """
    # Compose a summary of sources for the prompt
    source_summary = ""
    for i, src in enumerate(sources):
        source_summary += f"Source {i+1} ({src.get('source_url', 'N/A')}):\n{src.get('text', 'N/A')[:500]}...\n\n"
    outline_prompt = f"""
    Based on the following {len(sources)} collected source materials, generate a logical outline in {lang_code} for an academic paper on the topic: '{topic}'.
    Use standard outline formatting. Only output the outline.
    Collected Source Material (Summaries):\n---\n{source_summary[:10000]}\n---
    """
    return call_llm_api(outline_prompt, lang_code=lang_code, model=model, max_tokens=max_tokens)

# --- MCP Tool: synthesize_section ---
@mcp.tool(name="synthesize_section")
def synthesize_section(
    section_title: str,
    outline: str,
    sources: list,
    lang_code: str = 'en',
    max_tokens: int = 500,
    model: str = DEEPSEEK_MODEL
) -> str:
    """
    Synthesizes a section of an academic paper based on the section title, outline, and provided sources.
    Args:
        section_title (str): The section to write.
        outline (str): The full outline for context.
        sources (list): List of source dicts with 'text' and 'source_url'.
        lang_code (str): Output language code.
        max_tokens (int): Max tokens for LLM call.
        model (str): LLM model ID.
    Returns:
        str: The synthesized section content.
    """
    relevant_info_parts = []
    for idx, src in enumerate(sources):
        source_num = idx + 1
        url = src.get('source_url', 'N/A')
        text_snippet = src.get('text', 'N/A')[:1500]
        relevant_info_parts.append(f"Source {source_num}:\n{text_snippet}\n(URL: {url})")
    relevant_info_prompt_block = "\n\n".join(relevant_info_parts)
    synthesis_prompt = f"""
    Objective: Write a concise but informative section in {lang_code} covering: "{section_title}".
    Base your writing primarily on the provided source materials.
    Full Current Outline (for context ONLY):\n{outline}\n---
    Section to Write: {section_title}\n---
    Provided Source Materials (Use these to write the section):\n---\n{relevant_info_prompt_block[:10000]}\n---
    Instructions: Use citations as [Source N] where N is the source number. Respond in {lang_code}.
    """
    return call_llm_api(synthesis_prompt, lang_code=lang_code, model=model, max_tokens=max_tokens)

# --- MCP Tool: refine_search_query ---
@mcp.tool(name="refine_search_query")
def refine_search_query(
    failed_query: str,
    topic: str,
    lang_code: str = 'en',
    max_tokens: int = 100,
    model: str = DEEPSEEK_MODEL
) -> str:
    """
    Refines a failed search query for better results using the LLM.
    Args:
        failed_query (str): The previous search query that failed.
        topic (str): The research topic.
        lang_code (str): Output language code.
        max_tokens (int): Max tokens for LLM call.
        model (str): LLM model ID.
    Returns:
        str: The refined search query.
    """
    refinement_prompt = f"""
    The previous English Google search query '{failed_query}' for the overall topic '{topic}' failed to return useful results.
    Please generate one single, alternative English search query that is substantially different and might yield better results. Output only the new query string.
    """
    return call_llm_api(refinement_prompt, lang_code=lang_code, model=model, max_tokens=max_tokens)

# --- MCP Tool: format_citations ---
@mcp.tool(name="format_citations")
def format_citations(
    sources: list,
    style: str = 'APA',
    lang_code: str = 'en',
    max_tokens: int = 300,
    model: str = DEEPSEEK_MODEL
) -> str:
    """
    Formats a list of sources into a specified citation style using the LLM.
    Args:
        sources (list): List of source dicts with 'text' and 'source_url'.
        style (str): Citation style ('APA', 'MLA', etc.).
        lang_code (str): Output language code.
        max_tokens (int): Max tokens for LLM call.
        model (str): LLM model ID.
    Returns:
        str: The formatted citations.
    """
    citation_text = ""
    for i, src in enumerate(sources):
        citation_text += f"[{i+1}] {src.get('source_url', 'N/A')}\n"
    citation_prompt = f"""
    Format the following sources as a bibliography in {style} style. Only output the formatted citations.
    Sources:\n{citation_text}
    """
    return call_llm_api(citation_prompt, lang_code=lang_code, model=model, max_tokens=max_tokens)

# --- MCP Tool: plan_and_write_paper ---
@mcp.tool(name="plan_and_write_paper")
def plan_and_write_paper(
    topic: str,
    lang_code: str = 'en',
    target_sources: int = 40,
    max_searches: int = 10,
    study_mode: str = 'testing',
    max_refinements: int = 3,
    max_tokens_outline: int = 1500,
    max_tokens_synthesis: int = 500,
    search_results_per_q: int = 5,
    max_tokens_query_gen: int = 300,
    max_tokens_query_refine: int = 100,
    model: str = DEEPSEEK_MODEL
) -> dict:
    """
    Full workflow: plans and writes an academic paper on the given topic.
    Args: (see DS_Deepresearch.py for details)
    Returns:
        dict: Contains topic, sources, outline, synthesis, and formatted citations.
    """
    # Step 1: Generate search queries
    query_gen_prompt = f"""
    Based on the research topic "{topic}", generate a list of 5 diverse and optimized English Google search queries designed to find the most relevant and high-quality sources (like academic papers, reputable web resources).
    Focus on precision, key technical terms, synonyms, and potential variations that capture the core concepts effectively for maximizing relevant search results.
    Output ONLY the list of queries, one query per line. Do not add numbering or introductory text.
    """
    llm_generated_queries_str = call_llm_api(query_gen_prompt, lang_code='en', model=model, max_tokens=max_tokens_query_gen)
    search_queries = [q.strip() for q in llm_generated_queries_str.splitlines() if q.strip()]
    if not search_queries:
        search_queries = [f'"{topic}"', f'"{topic}" review', f'"{topic}" survey']
    gathered_sources_dict = {}
    query_attempts = 0
    refinement_attempts = 0
    query_index = 0
    while len(gathered_sources_dict) < target_sources and query_attempts < max_searches:
        current_query = search_queries[query_index % len(search_queries)]
        query_attempts += 1
        search_result = perform_external_search(current_query, num_results=search_results_per_q)
        search_succeeded = False
        if search_result and search_result.get('source_url'):
            source_url = search_result['source_url']
            if "google.com/search" not in source_url and "bing.com/search" not in source_url:
                if source_url not in gathered_sources_dict:
                    gathered_text = search_result.get('text', f"Snippet unavailable for {source_url}")
                    gathered_sources_dict[source_url] = {'text': gathered_text, 'source_url': source_url}
                    search_succeeded = True
        if not search_succeeded and refinement_attempts < max_refinements and query_attempts < max_searches:
            refinement_attempts += 1
            new_query_str = refine_search_query(current_query, topic, lang_code='en', model=model, max_tokens=max_tokens_query_refine)
            if new_query_str and len(new_query_str) > 3:
                search_queries[query_index % len(search_queries)] = new_query_str.strip('"\' \n\t')
            else:
                query_index += 1
        else:
            query_index += 1
    sources = list(gathered_sources_dict.values())
    # Step 2: Generate outline
    outline = generate_outline(topic, sources, lang_code=lang_code, max_tokens=max_tokens_outline, model=model)
    # Step 3: Synthesize sections
    def parse_outline_sections(outline_text):
        import re
        pattern = r"^\s*(?:[IVXLCDM]+\.|[A-Z]\.|[0-9]+\.|[a-z]\.|[一二三四五六七八九十]+[、．\.])\s+.*"
        sections = []
        if not isinstance(outline_text, str): return sections
        lines = outline_text.splitlines()
        for line in lines:
            trimmed_line = line.strip()
            if re.match(pattern, trimmed_line):
                parts = trimmed_line.split(None, 1)
                if (len(parts) > 1 and len(parts[1].strip()) > 0) or (len(parts) == 1 and len(parts[0]) > 5):
                    sections.append(trimmed_line)
        if not sections:
            sections = [line.strip() for line in lines if line.strip() and len(line.strip()) > 1]
        return sections
    current_outline_sections = parse_outline_sections(outline)
    if study_mode == 'testing':
        sections_to_synthesize = current_outline_sections[:2]
    else:
        sections_to_synthesize = current_outline_sections
    synthesis = {}
    for section_id in sections_to_synthesize:
        synthesized_content = synthesize_section(section_id, outline, sources, lang_code=lang_code, max_tokens=max_tokens_synthesis, model=model)
        synthesis[section_id] = synthesized_content
    # Step 4: Format citations
    formatted_citations = format_citations(sources, style='APA', lang_code=lang_code, model=model)
    return {
        "topic": topic,
        "sources": sources,
        "outline": outline,
        "synthesis": synthesis,
        "citations": formatted_citations
    }

if __name__ == "__main__":
    # Validate essential environment variable
    if not OPENROUTER_API_KEY:
        print("Missing OPENROUTER_API_KEY environment variable. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Start the MCP server with stdio transport
    print("Starting MCP Server in stdio mode", file=sys.stderr)
    mcp.run(transport="stdio")
