#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Pengwei Zhu
# Licensed under the MIT License. See README or LICENSE file for details.

import requests
import json
import os
import time
import re
import requests.exceptions
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import threading
import queue # For thread-safe GUI updates
# NOTE: No 'google_search' import needed here. External search implementation required by user.
# Potentially add imports for your chosen search library and PDF parsing library here
# Example:
# from googleapiclient.discovery import build # For Google Custom Search API
# import PyPDF2 # For PDF parsing
# from io import BytesIO # For handling downloaded PDF bytes
# from dotenv import load_dotenv # Import if using .env files

# --- Configuration ---
# Load environment variables (consider using dotenv)
# load_dotenv() # Uncomment this line if you want to load from a .env file

# OpenRouter Key (for LLM calls)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# SerpApi Key (for Web Search)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# *** Check for keys needed to run the GUI ***
# Keys are checked again before starting the actual research process
keys_ok = True
missing_keys_startup = []
if not OPENROUTER_API_KEY:
    missing_keys_startup.append("OPENROUTER_API_KEY")
    keys_ok = False
if not SERPAPI_API_KEY:
    # This is only a warning for startup, search function handles missing key later
    print("--- Startup Warning: SERPAPI_API_KEY not set. Web search will be skipped if not provided later. ---")
    # keys_ok = False # Don't block startup for SerpApi key, just warn later

# If essential keys are missing at startup, we might want to exit early or disable functionality
# For now, the GUI check handles this more gracefully.


# *** API and Model Configuration ***
DEEPSEEK_MODEL = "deepseek/deepseek-chat" # Ensure this matches OpenRouter ID
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_ENDPOINT = "https://serpapi.com/search" # SerpApi endpoint

# Headers for OpenRouter LLM calls
# Ensure OPENROUTER_API_KEY is loaded before defining this
LLM_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}", # Key loaded at start
    "Content-Type": "application/json",
    "HTTP-Referer": "YOUR_APP_URL_OR_NAME", # Optional: Replace with your app's URL or name
    "X-Title": "DSDeepResearch_Lite GUI" # Optional: Replace with your app's name
}

# --- Constants ---
# Token limits for API calls (Adjust as needed)
MAX_TOKENS_SYNTHESIS_PER_SECTION = 500 # Keep low for testing
MAX_TOKENS_OUTLINE = 1500
MAX_TOKENS_PLAN = 500
MAX_TOKENS_QUERY_GENERATION = 300
MAX_TOKENS_QUERY_REFINEMENT = 100

# Source Gathering Parameters
TARGET_SOURCE_COUNT = 40 # Target number of sources (may not be reached)
SEARCH_RESULTS_PER_QUERY = 5 # How many results to request from SerpApi per query
MAX_SEARCH_ATTEMPTS = 10 # Limit SerpApi calls
MAX_QUERY_REFINEMENT_ATTEMPTS = 3 # Limit how many times we ask LLM to refine within the loop

# --- Language Strings ---
# Dictionary holding UI text and prompt instructions for different languages
LANG_STRINGS = {
    'en': {
        'system_role': "You are an advanced research assistant responding in English.",
        'plan_instruction': "Please respond concisely in English.", # Plan is always English now
        'info_instruction': "Please provide the information concisely and in English.", # Info gathering (LLM fallback)
        'outline_instruction': "Please use standard English outline formatting (e.g., I., A., 1.). Please respond in English.", # Outline in selected lang
        'synthesis_instruction': "Write the content concisely and in English based *only* on the provided source materials for this section.", # Synthesis in selected lang
        'citation_instruction_en': """
            7. **Citation Instruction:** Reference facts using inline citations corresponding to the source number provided with the background information (e.g., [Source 1], [Source 5]). Accuracy is paramount.
            """,
        'citation_instruction_no_source_en': """
            7. **Citation Instruction:** No specific source material was provided for this section, rely on general knowledge if necessary but state that source material was missing. Do not add citations.
            """,
        'not_generated': "Not generated.",
        'content_warning': "(Content based on gathered sources via SerpApi - General Web Search)", # Updated warning
        'char_count_label': "Word count",
        'char_count_total_label': "Estimated total words so far",
        'final_char_count_label': "Final Estimated Word Count",
        'section_not_generated': "\n*Content not generated for this section.*\n",
        'search_fail_text': "SerpApi search failed or returned no results.",
        'references_heading': "Collected Sources (via SerpApi)",
        'no_references': "[No sources collected or SerpApi key missing/invalid]",
        'gui_topic_label': "Research Topic:",
        'gui_language_label': "Output Language:",
        'gui_start_button': "Start Research",
        'gui_status_ready': "Ready. Enter topic and language.",
        'gui_status_running': "Research in progress...",
        'gui_status_done': "Research process finished.",
        'gui_status_error': "Error occurred.",
        'gui_keys_missing_title': "API Keys Missing",
        'gui_keys_missing_msg': "Error: OPENROUTER_API_KEY environment variable not set.\nPlease set it (and SERPAPI_API_KEY for web search) before running.", # Simplified message
        'gui_serpapi_missing_warning': "WARNING: SerpApi key not provided. Web search step will be skipped.",
    },
    'zh': {
        'system_role': "You are an advanced research assistant responding in Chinese (简体中文).",
        'plan_instruction': "Please respond concisely in English.", # Plan is always English now
        'info_instruction': "请用简体中文简洁地提供信息。", # Info gathering (LLM fallback)
        'outline_instruction': "请使用标准中文大纲格式（例如：一、 (一) 1.）。请用简体中文回答。", # Outline in selected lang
        'synthesis_instruction': "请*仅*基于本节提供的源材料，用简体中文简洁地撰写内容。", # Synthesis in selected lang
        'citation_instruction_zh': """
            7. **引用说明:** 请使用背景信息中提供的来源编号对应的内联引文（例如 [来源 1], [来源 5]）来引用事实。准确性至关重要。
            """,
        'citation_instruction_no_source_zh': """
            7. **引用说明:** 本节未提供具体的来源材料，如有必要请依赖通用知识，但需说明来源材料缺失。请勿添加引文。
            """,
        'not_generated': "未生成。",
        'content_warning': "(内容基于通过SerpApi收集的来源 - 通用网络搜索)", # Updated warning
        'char_count_label': "字符数",
        'char_count_total_label': "累计估算字符数",
        'final_char_count_label': "最终估算字符数",
        'section_not_generated': "\n*此部分内容未生成。*\n",
        'search_fail_text': "SerpApi搜索失败或未返回结果。",
        'references_heading': "收集的来源 (通过 SerpApi)",
        'no_references': "[未收集到来源或SerpApi密钥丢失/无效]",
        'gui_topic_label': "研究主题:",
        'gui_language_label': "输出语言:",
        'gui_start_button': "开始研究",
        'gui_status_ready': "就绪。请输入主题和语言。",
        'gui_status_running': "研究进行中...",
        'gui_status_done': "研究过程结束。",
        'gui_status_error': "发生错误。",
        'gui_keys_missing_title': "API密钥缺失",
        'gui_keys_missing_msg': "错误：未设置 OPENROUTER_API_KEY 环境变量。\n请在运行前设置它（以及用于网络搜索的 SERPAPI_API_KEY）。", # Simplified message
        'gui_serpapi_missing_warning': "警告：未提供 SerpApi 密钥。将跳过网络搜索步骤。",
    }
}


# --- Core API Interaction Function (for LLM) ---
# (call_llm_api function remains the same as previous version)
def call_llm_api(prompt, lang_code='en', model=DEEPSEEK_MODEL, max_tokens=2000, temperature=0.7):
    """
    Sends a prompt to the configured LLM API (via OpenRouter) and returns the response content.
    Handles API key check and basic errors.

    Args:
        prompt (str): The user prompt for the LLM.
        lang_code (str): The target language code ('en' or 'zh').
        model (str): The specific model identifier for the API.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The sampling temperature for generation.

    Returns:
        str: The content generated by the LLM, or an error string starting with "Error:".
    """
    if not OPENROUTER_API_KEY: return "Error: OPENROUTER_API_KEY not configured."
    system_message_base = LANG_STRINGS[lang_code]['system_role']
    system_message = f"""
    {system_message_base} Your primary goals are:
    1. Accuracy & Detail
    2. Instruction Adherence (including language: {lang_code})
    3. Consistency
    4. Clarity
    5. Conditional Citation (follow instructions in user prompt)
    """
    payload = { "model": model, "messages": [{"role": "system", "content": system_message.strip()}, {"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature }
    print(f"\n--- Sending Prompt to LLM {model} via OpenRouter (Lang: {lang_code}) ---")
    print(f"Prompt (first 300 chars): {prompt[:300]}...")
    print(f"(Requesting max_tokens: {max_tokens})")
    response = None
    try:
        print(f"--- Making requests.post call to {OPENROUTER_API_URL} (Timeout={180}s) ---")
        current_llm_headers = LLM_HEADERS.copy()
        current_llm_headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        response = requests.post( OPENROUTER_API_URL, headers=current_llm_headers, json=payload, timeout=180 )
        print(f"--- requests.post call returned (Status: {response.status_code}) ---")
        if response.status_code != 200:
            print(f"Error Response Text: {response.text}")
            response.raise_for_status()
        result = response.json()
        if "choices" in result and result["choices"] and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
            content = result["choices"][0]["message"]["content"]
            if "usage" in result: print(f"--- API Usage: {result['usage']} ---")
            return content.strip()
        else:
            print("Warning: API response status 200 but did not contain expected content path.")
            print("Full response:", json.dumps(result, indent=2))
            if "error" in result:
                 error_message = result['error'].get('message', 'No message provided')
                 try:
                     if '\\u' in error_message: error_message = error_message.encode('utf-8').decode('unicode_escape')
                 except Exception: pass
                 print(f"API Error Message: {error_message}")
                 return f"Error: API returned an error - {result['error'].get('code', 'Unknown code')}"
            return "Error: Unexpected API response format"
    except requests.exceptions.Timeout:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error: LLM API request timed out after 180 seconds.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error: LLM API request failed: {e}")
        error_details = ""
        error_text = ""
        if response is not None:
             try:
                error_details = response.json()
                if "error" in error_details:
                     err_data = error_details["error"]
                     print(f"API Error Code: {err_data.get('code', 'N/A')}")
                     print(f"API Error Message: {err_data.get('message', 'N/A')}")
                     if err_data.get('code') == "invalid_api_key": print("Error Detail: Invalid API Key.")
                     elif "context_length_exceeded" in err_data.get("code", ""): print("Error Detail: Context length exceeded.")
                elif "rate limit" in str(error_details).lower(): print("Error Detail: Rate limit likely exceeded.")
             except json.JSONDecodeError:
                error_text = response.text
                error_details = error_text
                if "rate limit" in error_text.lower(): print("Error Detail: Rate limit likely exceeded.")
        print(f"Full Error details (if available): {error_details}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return f"Error: API request failed ({e})"
    except json.JSONDecodeError:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Error: Could not decode JSON response from LLM API, even though status was OK.")
        print("Response text:", response.text if response is not None else "N/A")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return "Error: Invalid JSON response"
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"An unexpected error occurred during LLM API call: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return f"Error: An unexpected error occurred ({e})."


# --- Helper Function (Implemented External Search using SerpApi) ---
# (perform_external_search function remains the same)
def perform_external_search(query: str):
    """
    Performs a general web search using the SerpApi Google Search API.
    Uses hl=en for searching. Returns {'text': ..., 'source_url': ...} or None.
    """
    if not SERPAPI_API_KEY:
        print("--- SerpApi API Key not provided. Skipping web search. ---")
        return None
    hl_param = 'en'
    params = { "q": query, "api_key": SERPAPI_API_KEY, "num": SEARCH_RESULTS_PER_QUERY, "hl": hl_param, "engine": "google" }
    print(f"--- Calling SerpApi Search ---")
    print(f"--- Query: {query} ---")
    print(f"--- Lang (hl): {hl_param} ---")
    response = None
    try:
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
            if "error" in results: print(f"--- SerpApi returned an error message: {results['error']} ---")
            return None
    except requests.exceptions.Timeout:
        print("--- Error: SerpApi request timed out. ---")
        return None
    except requests.exceptions.RequestException as e:
        print(f"--- Error: SerpApi request failed: {e} ---")
        if hasattr(e, 'response') and e.response is not None:
             print(f"SerpApi Error Response Status Code: {e.response.status_code}")
             print(f"SerpApi Error Response Body: {e.response.text}")
        return None
    except json.JSONDecodeError:
        print("--- Error: Failed to decode JSON response from SerpApi. ---")
        print("Raw Response:", response.text if response is not None else "N/A")
        return None
    except Exception as e:
        print(f"--- An unexpected error occurred during SerpApi search: {e} ---")
        return None


# --- Helper Function (Placeholder for PDF Text Extraction) ---
# (extract_text_from_pdf function remains the same - placeholder)
def extract_text_from_pdf(pdf_url: str):
    """
    *** USER IMPLEMENTATION REQUIRED ***
    Downloads a PDF and extracts text. Requires 'requests' and 'PyPDF2' (or similar).
    Returns text or None.
    """
    print(f"--- [Placeholder] Attempting to download and extract text from PDF: {pdf_url} ---")
    print("--- PDF Extraction not implemented. Returning None. ---")
    return None


# --- Helper Function (Outline Parsing - Needs Improvement) ---
# (parse_outline_sections function remains the same)
def parse_outline_sections(outline_text):
    """
    Parses outline text to extract section identifiers/titles.
    Uses regex for common English and Chinese formats. Needs improvement for robustness.
    """
    pattern = r"^\s*(?:[IVXLCDM]+\.|[A-Z]\.|[0-9]+\.|[a-z]\.|[一二三四五六七八九十]+[、．\.])\s+.*"
    sections = []
    lines = outline_text.splitlines()
    for line in lines:
        trimmed_line = line.strip()
        if re.match(pattern, trimmed_line):
             parts = trimmed_line.split(None, 1)
             if (len(parts) > 1 and len(parts[1].strip()) > 0) or \
                (len(parts) == 1 and len(parts[0]) > 5):
                  sections.append(trimmed_line)
    if not sections:
         sections = [line.strip() for line in lines if line.strip() and len(line.strip()) > 1]
         print("Warning: Basic outline parsing failed to find standard patterns. Using non-empty lines as sections.")
    print(f"Parsed {len(sections)} potential sections from outline using basic regex.")
    return sections


# --- Research Workflow Simulation (Source-First with SerpApi, General Search) ---
# (run_research_process function remains the same, including query refinement logic)
def run_research_process(topic, lang_code='en', output_queue=None):
    """
    Main research process function, adapted for GUI. Runs the source-first workflow.
    Includes LLM query generation and refinement.
    """
    def log_message(message):
        if output_queue: output_queue.put(str(message))
        else: print(message)

    log_message(f"\n=== Starting Research Process for: {topic} (Language: {lang_code}, Workflow: Source-First/SerpApi General Search) ===")
    lang_strings = LANG_STRINGS[lang_code]
    research_data = { "topic": topic, "sources": [], "outline": "", "synthesis": {}, "estimated_char_count": 0 }
    target_model_id = DEEPSEEK_MODEL
    log_message(f"--- Using LLM Model ID: {target_model_id} ---")

    # --- Step 1: Source Gathering ---
    log_message(f"\n[Step 1: Gathering Sources via SerpApi (Max Attempts: {MAX_SEARCH_ATTEMPTS}, Search lang: en)]")
    log_message("--- Generating OPTIMIZED search queries using LLM ---")
    query_gen_prompt = f"""
    Based on the research topic "{topic}", generate a list of 5 diverse and optimized English Google search queries designed to find the most relevant and high-quality sources (like academic papers, reputable web resources).
    Focus on precision, key technical terms, synonyms, and potential variations that capture the core concepts effectively for maximizing relevant search results.
    Output ONLY the list of queries, one query per line. Do not add numbering or introductory text.
    """
    llm_generated_queries_str = call_llm_api(query_gen_prompt, lang_code='en', model=target_model_id, max_tokens=MAX_TOKENS_QUERY_GENERATION)
    search_queries = []
    if not llm_generated_queries_str.startswith("Error:") and llm_generated_queries_str:
        search_queries = [q.strip() for q in llm_generated_queries_str.splitlines() if q.strip()]
        search_queries = [q for q in search_queries if len(q) > 3 and not q.startswith("Here") and not q.startswith("1.")]
        log_message(f"--- LLM generated {len(search_queries)} initial search queries: ---")
        for q in search_queries: log_message(f"  - {q}")
    else:
        log_message("--- Warning: Failed to generate search queries via LLM. Using basic topic query. ---")
        search_queries = []
    if not search_queries:
         log_message("--- Using fallback basic queries ---")
         search_queries = [f'"{topic}"', f'"{topic}" review', f'"{topic}" survey']

    gathered_sources_dict = {}
    query_attempts = 0
    refinement_attempts = 0
    max_query_attempts = MAX_SEARCH_ATTEMPTS
    query_index = 0

    while len(gathered_sources_dict) < TARGET_SOURCE_COUNT and query_attempts < max_query_attempts:
        if not search_queries:
             log_message("--- ERROR: No valid search queries available. Stopping search. ---")
             break
        current_query = search_queries[query_index % len(search_queries)]
        query_attempts += 1
        log_message(f"\n--- Search Attempt {query_attempts}/{max_query_attempts}: Query: '{current_query}' ---")
        search_result = perform_external_search(current_query)
        search_succeeded = False
        if search_result and search_result.get('source_url'):
            source_url = search_result['source_url']
            if "google.com/search" not in source_url and "bing.com/search" not in source_url:
                if source_url not in gathered_sources_dict:
                    gathered_text = search_result.get('text', f"Snippet unavailable for {source_url}")
                    # pdf_text = extract_text_from_pdf(source_url) # Placeholder
                    # if pdf_text: gathered_text = f"Extracted PDF Text (Partial): {pdf_text}"
                    gathered_sources_dict[source_url] = {'text': gathered_text, 'source_url': source_url}
                    log_message(f"--- Collected source {len(gathered_sources_dict)}/{TARGET_SOURCE_COUNT}: {source_url} ---")
                    log_message(f"    Snippet: {gathered_text[:100]}...")
                    search_succeeded = True
                else:
                    log_message(f"--- Skipping duplicate source: {source_url} ---")
                    search_succeeded = True
            else:
                 log_message(f"--- Skipping search engine result page: {source_url} ---")
                 search_succeeded = False
        else:
            log_message(f"--- Search attempt yielded no usable result or URL. ---")
            if search_result is None and SERPAPI_API_KEY:
                 log_message("--- Search failed. Check SerpApi key and service status. ---")
            search_succeeded = False

        if not search_succeeded and refinement_attempts < MAX_QUERY_REFINEMENT_ATTEMPTS and query_attempts < max_query_attempts:
            refinement_attempts += 1
            log_message(f"--- Search failed. Attempting LLM query refinement ({refinement_attempts}/{MAX_QUERY_REFINEMENT_ATTEMPTS}) ---")
            refinement_prompt = f"""
            The previous English Google search query '{current_query}' for the overall topic '{topic}' failed to return useful results.
            Please generate one single, alternative English search query that is substantially different (e.g., broader, narrower, different keywords, different angle) and might yield better results for finding relevant academic papers or high-quality web resources.
            Output only the new query string, and nothing else.
            """
            new_query_str = call_llm_api(refinement_prompt, lang_code='en', model=target_model_id, max_tokens=MAX_TOKENS_QUERY_REFINEMENT)
            if not new_query_str.startswith("Error:") and len(new_query_str) > 3:
                 new_query_str = new_query_str.strip('\'" \n\t')
                 log_message(f"--- LLM suggested new query: '{new_query_str}' ---")
                 search_queries[query_index % len(search_queries)] = new_query_str
            else:
                 log_message("--- LLM failed to generate refinement query. Continuing with next original query. ---")
                 query_index += 1
        else:
             query_index += 1

        if len(gathered_sources_dict) >= TARGET_SOURCE_COUNT:
            log_message(f"--- Reached target source count ({TARGET_SOURCE_COUNT}). Stopping search. ---")
            break
        time.sleep(1.5)

    research_data["sources"] = list(gathered_sources_dict.values())
    log_message(f"\n--- Finished Source Gathering after {query_attempts} attempts: Collected {len(research_data['sources'])} unique sources. ---")
    if not research_data["sources"]:
        log_message("Could not gather any sources. Unable to proceed.")
        output_queue.put("===RESEARCH_ERROR===")
        return None

    source_summary_for_outline = ""
    for i, src in enumerate(research_data["sources"]):
        source_summary_for_outline += f"Source {i+1} ({src.get('source_url', 'N/A')}):\n{src.get('text', 'N/A')[:500]}...\n\n"

    # --- Step 2: Outline Generation ---
    # (Outline generation logic remains the same)
    log_message(f"\n[Step 2: Generating Outline from Sources ({lang_code})]")
    outline_prompt = f"""
    Based on the following {len(research_data['sources'])} collected source materials (Titles/Snippets from Web Search), generate a logical, hierarchical outline in {lang_code} for a research paper on the topic: '{topic}'.
    The outline should synthesize the key themes, findings, and structure suggested by these sources.
    Aim for a structure suitable for a report based on these sources. Use standard outline formatting appropriate for the language ({lang_code}).
    {LANG_STRINGS[lang_code]['outline_instruction']}

    Collected Source Material (Summaries):
    ---
    {source_summary_for_outline[:10000]}
    ---
    """
    research_data["outline"] = call_llm_api(outline_prompt, lang_code=lang_code, model=target_model_id, max_tokens=MAX_TOKENS_OUTLINE)
    if research_data["outline"].startswith("Error:"):
        log_message(f"Failed to generate outline from sources ({lang_code}). Exiting.")
        output_queue.put("===RESEARCH_ERROR===")
        return None
    log_message(f"\nGenerated Outline (Based on Sources, {lang_code}):")
    log_message(research_data["outline"])
    time.sleep(1)


    # --- Step 3: Section-by-Section Synthesis ---
    # (Synthesis logic remains the same)
    log_message(f"\n[Step 3: Synthesizing Content Section by Section ({lang_code})]")
    current_outline_sections = parse_outline_sections(research_data["outline"])
    SECTIONS_TO_PROCESS_LIMIT = 3
    sections_to_synthesize = current_outline_sections[:SECTIONS_TO_PROCESS_LIMIT]
    log_message(f"--- Processing only the first {len(sections_to_synthesize)} outline points for synthesis (Testing Mode) ---")

    if not sections_to_synthesize:
        log_message("Could not parse sections from the outline or limit resulted in zero sections. Cannot synthesize content.")
    else:
        total_synthesized_chars = 0
        for i, section_id in enumerate(sections_to_synthesize):
            log_message(f"\n--- Synthesizing section {i+1}/{len(sections_to_synthesize)}: '{section_id}' ---")
            relevant_info_parts = []
            source_mapping = {}
            for idx, src in enumerate(research_data["sources"]):
                 source_num = idx + 1
                 url = src.get('source_url', 'N/A')
                 text_snippet = src.get('text', 'N/A')[:1500]
                 relevant_info_parts.append(f"Source {source_num}:\n{text_snippet}\n(URL: {url})")
                 source_mapping[source_num] = url
            relevant_info_prompt_block = "\n\n".join(relevant_info_parts)
            if research_data["sources"]: citation_instruction = lang_strings[f'citation_instruction_{lang_code}']
            else: citation_instruction = lang_strings[f'citation_instruction_no_source_{lang_code}']
            synthesis_prompt = f"""
            Objective: Write a **concise but informative** section in {lang_code} for the research document on '{research_data['topic']}', exclusively covering the content for the outline point: "{section_id}".
            Base your writing *primarily* on the provided source materials gathered from web search.
            **Full Current Outline (for context ONLY):**\n{research_data['outline']}\n---
            **Section to Write:** {section_id}\n---
            **Provided Source Materials (Use these to write the section):**\n---\n{relevant_info_prompt_block[:10000]}\n---
            Instructions:
            1. Write concise, coherent paragraphs covering the specific topic of section "{section_id}".
            2. Synthesize information *from the provided source materials*. Refer to them using inline citations like [Source 1], [Source 5], etc., corresponding to the numbers provided with each source snippet above. Be accurate.
            3. Ensure the content is logically consistent with the outline and the sources.
            4. **Do NOT write content for any other outline sections**.
            5. Maintain a clear, objective, and formal style.
            6. Generate concise text for this section.
            {citation_instruction}
            {lang_strings['synthesis_instruction']}
            """
            synthesized_content = call_llm_api(synthesis_prompt, lang_code=lang_code, model=target_model_id, max_tokens=MAX_TOKENS_SYNTHESIS_PER_SECTION)
            if not synthesized_content.startswith("Error:"):
                research_data["synthesis"][section_id] = synthesized_content
                section_char_count = len(synthesized_content)
                total_synthesized_chars += section_char_count
                log_message(f"\nSynthesized Content for '{section_id}' ({lang_strings['char_count_label']}: {section_char_count}):")
                log_message(synthesized_content[:300] + "...")
            else:
                log_message(f"Failed to synthesize content for section '{section_id}'.")
                research_data["synthesis"][section_id] = f"Error: Failed to generate content for {section_id}."
            research_data["estimated_char_count"] = total_synthesized_chars
            log_message(f"{lang_strings['char_count_total_label']}: {total_synthesized_chars}")
            time.sleep(2)

    log_message(f"\n=== Research Process for: {topic} Completed (Source-First Mode) ===")
    log_message(f"Final Estimated Character Count (Synthesized Content): {research_data['estimated_char_count']}")

    # --- Assemble Final Document ---
    # (Assembly logic remains the same)
    lang_strings = LANG_STRINGS[lang_code]
    final_content_warning = "(Content based on gathered sources via SerpApi - General Search)"
    if not SERPAPI_API_KEY or not research_data.get("sources"): final_content_warning = "(Web search skipped or failed; content may be limited)"
    full_text = []
    full_text.append(f"# Research Document: {research_data['topic']}\n")
    full_text.append(f"\n## Outline (Generated from Sources) ({lang_code.upper()})")
    full_text.append(research_data.get('outline', lang_strings['not_generated']))
    full_text.append(f"\n## Synthesized Content {final_content_warning}")
    processed_sections = list(research_data['synthesis'].keys())
    final_outline_sections = parse_outline_sections(research_data['outline'])
    ordered_sections_to_print = [s for s in final_outline_sections if s in processed_sections]
    for section_key in processed_sections:
         if section_key not in ordered_sections_to_print: ordered_sections_to_print.append(section_key)
    if ordered_sections_to_print:
         for section in ordered_sections_to_print:
              content = research_data['synthesis'].get(section, lang_strings['section_not_generated'])
              full_text.append(f"\n### {section.strip()}")
              full_text.append(content)
         if len(ordered_sections_to_print) < len(final_outline_sections):
              full_text.append(f"\n\n[Note: Only the first {len(ordered_sections_to_print)} sections were processed in this short run.]")
    else: full_text.append("\n[No sections were synthesized in this run.]")
    full_text.append(f"\n\n## {lang_strings['references_heading']}\n")
    if research_data.get("sources"):
        sorted_sources = sorted(research_data["sources"], key=lambda x: x.get('source_url', ''))
        for i, src in enumerate(sorted_sources):
            url = src.get('source_url', 'N/A')
            text_snippet = src.get('text', '')[:120]
            full_text.append(f"{i+1}. URL: {url}\n   Info: {text_snippet}...")
    else: full_text.append(lang_strings['no_references'])
    assembled_document = "\n".join(full_text)

    # Put final result in queue for GUI
    if output_queue:
        output_queue.put("\n\n=== Final Assembled Document ===\n")
        output_queue.put(assembled_document)
    else:
        print("\n\n=== Final Assembled Document ===\n")
        print(assembled_document)

    return research_data # Return data


# --- GUI Application Class ---
# (GUI Class ResearchApp remains the same)
class ResearchApp:
    def __init__(self, root):
        """Initializes the Tkinter GUI application."""
        self.root = root
        self.root.title("DSDeepResearch_Lite GUI")
        self.root.minsize(600, 500)
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=10)
        text_font = font.nametofont("TkTextFont")
        text_font.configure(size=10)
        self.root.option_add("*Font", default_font)
        self.topic_var = tk.StringVar()
        self.lang_var = tk.StringVar(value='en')
        self.status_var = tk.StringVar(value="Initializing...")
        self.output_queue = queue.Queue()
        self.create_widgets()
        self.check_api_keys()

    def create_widgets(self):
        """Creates and lays out the GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        topic_label = ttk.Label(input_frame, text="Research Topic:")
        topic_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.topic_entry = ttk.Entry(input_frame, textvariable=self.topic_var, width=60)
        self.topic_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        lang_label = ttk.Label(input_frame, text="Output Language:")
        lang_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        lang_frame = ttk.Frame(input_frame)
        lang_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        en_radio = ttk.Radiobutton(lang_frame, text="English", variable=self.lang_var, value='en')
        en_radio.pack(side=tk.LEFT, padx=5)
        zh_radio = ttk.Radiobutton(lang_frame, text="Chinese (简体)", variable=self.lang_var, value='zh')
        zh_radio.pack(side=tk.LEFT, padx=5)

        self.start_button = ttk.Button(input_frame, text="Start Research", command=self.start_research_thread)
        self.start_button.grid(row=2, column=0, columnspan=2, pady=10)

        output_frame = ttk.LabelFrame(main_frame, text="Output Log & Results", padding="10")
        output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.rowconfigure(1, weight=1)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=20, width=80, state='disabled')
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        # --- Status Bar with Copyright ---
        status_bar = ttk.Frame(main_frame, relief=tk.SUNKEN, padding="2 5 2 5")
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=(5,0))

        self.status_label = ttk.Label(status_bar, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True) # Status fills available space

        # Add copyright label to the right
        copyright_text = "Copyright (c) 2025 Pengwei Zhu"
        self.copyright_label = ttk.Label(status_bar, text=copyright_text, anchor=tk.E)
        self.copyright_label.pack(side=tk.RIGHT) # Pack copyright to the right

    def log_to_gui(self, message):
        """Appends a message to the output text area in a thread-safe way."""
        try:
            self.output_text.configure(state='normal')
            self.output_text.insert(tk.END, str(message) + "\n")
            self.output_text.configure(state='disabled')
            self.output_text.see(tk.END)
            self.root.update_idletasks()
        except tk.TclError as e:
             print(f"GUI Error logging message: {e}")

    def check_api_keys(self):
        """Checks for API keys and updates status/button."""
        lang = self.lang_var.get()
        lang_strings = LANG_STRINGS[lang]
        if not OPENROUTER_API_KEY:
            error_msg = lang_strings['gui_keys_missing_msg']
            self.status_var.set(error_msg)
            self.log_to_gui(error_msg)
            self.start_button.configure(state='disabled')
            messagebox.showerror(lang_strings['gui_keys_missing_title'], error_msg)
            return False
        else:
            if not SERPAPI_API_KEY:
                 self.status_var.set(lang_strings['gui_serpapi_missing_warning'])
                 self.log_to_gui(lang_strings['gui_serpapi_missing_warning'])
            else:
                 self.status_var.set(lang_strings['gui_status_ready'])
            self.start_button.configure(state='normal')
            return True

    def start_research_thread(self):
        """Starts the research process in a separate thread."""
        if not self.check_api_keys(): return
        topic = self.topic_var.get().strip()
        lang = self.lang_var.get()
        lang_strings = LANG_STRINGS[lang]
        if not topic:
            messagebox.showwarning("Input Missing", "Please enter a research topic.")
            return

        self.start_button.configure(state='disabled')
        self.output_text.configure(state='normal')
        self.output_text.delete('1.0', tk.END)
        self.output_text.configure(state='disabled')
        self.status_var.set(lang_strings['gui_status_running'])
        self.log_to_gui(f"Starting research for topic: '{topic}' in language: {lang}")
        if not SERPAPI_API_KEY: self.log_to_gui(lang_strings['gui_serpapi_missing_warning'])

        self.research_thread = threading.Thread( target=research_worker, args=(topic, lang, self.output_queue), daemon=True )
        self.research_thread.start()
        self.root.after(100, self.process_queue)

    def process_queue(self):
        """Processes messages from the research thread queue."""
        try:
            while True:
                message = self.output_queue.get_nowait()
                lang = self.lang_var.get()
                lang_strings = LANG_STRINGS[lang]
                if message == "===RESEARCH_COMPLETE===":
                    self.status_var.set(lang_strings['gui_status_done'])
                    self.start_button.configure(state='normal')
                elif message == "===RESEARCH_ERROR===":
                     self.status_var.set(lang_strings['gui_status_error'])
                     self.start_button.configure(state='normal')
                else:
                    self.log_to_gui(message)
        except queue.Empty:
            pass

        if hasattr(self, 'research_thread') and self.research_thread.is_alive():
            self.root.after(200, self.process_queue)
        elif hasattr(self, 'research_thread') and not self.research_thread.is_alive():
             try:
                  while True:
                       message = self.output_queue.get_nowait()
                       lang = self.lang_var.get()
                       lang_strings = LANG_STRINGS[lang]
                       if message == "===RESEARCH_COMPLETE===": self.status_var.set(lang_strings['gui_status_done'])
                       elif message == "===RESEARCH_ERROR===": self.status_var.set(lang_strings['gui_status_error'])
                       else: self.log_to_gui(message)
             except queue.Empty: pass
             if self.start_button['state'] == 'disabled':
                  self.start_button.configure(state='normal')
                  current_status = self.status_var.get()
                  if lang_strings['gui_status_done'] not in current_status and lang_strings['gui_status_error'] not in current_status:
                       self.status_var.set(lang_strings['gui_status_done'] + " (Thread ended)")


# --- Worker Function (Runs in Background Thread) ---
# (research_worker function remains the same)
import sys
def research_worker(topic, lang_code, output_queue):
    """
    Wrapper function to run the research process and put output/status in the queue.
    This function runs in the background thread. Redirects print to queue.
    """
    class QueueLogger:
        def __init__(self, queue): self.queue = queue
        def write(self, message):
            if message and not message.isspace(): self.queue.put(message.rstrip())
        def flush(self): pass

    original_stdout = sys.stdout
    sys.stdout = QueueLogger(output_queue)
    final_data = None
    try:
        final_data = run_research_process(topic, lang_code, output_queue)
        if final_data: output_queue.put("===RESEARCH_COMPLETE===")
        else:
            output_queue.put("Research process returned no data (likely due to an earlier error).")
            output_queue.put("===RESEARCH_ERROR===")
    except Exception as e:
        output_queue.put(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        output_queue.put(f"An error occurred in the research worker thread: {e}")
        import traceback
        output_queue.put(traceback.format_exc())
        output_queue.put(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        output_queue.put("===RESEARCH_ERROR===")
    finally:
        sys.stdout = original_stdout # Restore stdout
        # Optional: Trigger file saving from main thread if needed


# --- Main Execution ---
if __name__ == "__main__":
    # Sets up and runs the GUI
    root = tk.Tk()
    app = ResearchApp(root)
    root.mainloop() # Start the Tkinter event loop
