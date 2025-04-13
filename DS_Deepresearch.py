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
from tkinter import ttk, scrolledtext, messagebox, font, filedialog
import threading
import queue  # For thread-safe GUI updates
import sys  # Needed for stdout redirection in worker
import traceback  # Needed for error logging in worker
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
keys_ok = True
missing_keys_startup = []
if not OPENROUTER_API_KEY:
    missing_keys_startup.append("OPENROUTER_API_KEY")
    keys_ok = False
if not SERPAPI_API_KEY:
    print("--- Startup Warning: SERPAPI_API_KEY not set. Web search will be skipped if not provided later. ---")

# *** API and Model Configuration ***
# Default values
DEEPSEEK_MODEL = "deepseek/deepseek-chat"  # Fixed LLM Model ID - CORRECT VARIABLE NAME
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_ENDPOINT = "https://serpapi.com/search"  # SerpApi endpoint

# Headers for OpenRouter LLM calls
# Ensure OPENROUTER_API_KEY is loaded before defining this
LLM_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",  # Key loaded at start
    "Content-Type": "application/json",
    "HTTP-Referer": "YOUR_APP_URL_OR_NAME",  # Optional: Replace
    "X-Title": "DSDeepResearch_Lite GUI"  # Optional: Replace
}

# --- Default Constants (used to initialize GUI and as fallbacks) ---
# Token limits for API calls
DEFAULT_MAX_TOKENS_SYNTHESIS_PER_SECTION = 500  # Default for GUI
DEFAULT_MAX_TOKENS_OUTLINE = 1500              # Default for GUI
DEFAULT_MAX_TOKENS_QUERY_GENERATION = 300      # Default for GUI
DEFAULT_MAX_TOKENS_QUERY_REFINEMENT = 100      # Default for GUI

# Source Gathering Parameters
DEFAULT_TARGET_SOURCE_COUNT = 40  # Default for GUI
DEFAULT_SEARCH_RESULTS_PER_QUERY = 5  # Default for GUI
DEFAULT_MAX_SEARCH_ATTEMPTS = 10  # Default for GUI
DEFAULT_MAX_QUERY_REFINEMENT_ATTEMPTS = 3  # Default for GUI

# Default output path
DEFAULT_OUTPUT_PATH = os.path.join(os.getcwd(), "Deepresearch_Output")  # Default for GUI

# --- Language Strings ---
# (LANG_STRINGS dictionary remains the same as previous version)
LANG_STRINGS = {
    'en': {
        'system_role': "You are an advanced research assistant responding in English.",
        'plan_instruction': "Please respond concisely in English.",
        'info_instruction': "Please provide the information concisely and in English.",
        'outline_instruction': "Please use standard English outline formatting (e.g., I., A., 1.). Please respond in English.",
        'synthesis_instruction': "Write the content concisely and in English based *only* on the provided source materials for this section.",
        'citation_instruction_en': """
            7. **Citation Instruction:** Reference facts using inline citations corresponding to the source number provided with the background information (e.g., [Source 1], [Source 5]). Accuracy is paramount.
            """,
        'citation_instruction_no_source_en': """
            7. **Citation Instruction:** No specific source material was provided for this section, rely on general knowledge if necessary but state that source material was missing. Do not add citations.
            """,
        'not_generated': "Not generated.",
        'content_warning': "(Content based on gathered sources via SerpApi - General Web Search)",
        'char_count_label': "Word count",
        'char_count_total_label': "Estimated total words so far",
        'final_char_count_label': "Final Estimated Word Count",
        'section_not_generated': "\n*Content not generated for this section.*\n",
        'search_fail_text': "SerpApi search failed or returned no results.",
        'references_heading': "Collected Sources (via SerpApi)",
        'no_references': "[No sources collected or SerpApi key missing/invalid]",
        # GUI Labels & Tooltips
        'gui_topic_label': "Research Topic:",
        'gui_topic_tooltip': "Enter the main topic or research question.",
        'gui_language_label': "Output Language:",
        'gui_language_tooltip': "Select the language for the final report outline and content.",
        'gui_start_button': "Start Research",
        'gui_status_ready': "Ready. Enter topic and language.",
        'gui_status_running': "Research in progress...",
        'gui_status_done': "Research process finished.",
        'gui_status_error': "Error occurred.",
        'gui_keys_missing_title': "API Keys Missing",
        'gui_keys_missing_msg': "Error: OPENROUTER_API_KEY environment variable not set.\nPlease set it (and SERPAPI_API_KEY for web search) before running.",
        'gui_serpapi_missing_warning': "WARNING: SerpApi key not provided. Web search step will be skipped.",
        'gui_target_sources_label': "Target Sources:",
        'gui_target_sources_tooltip': "Desired number of unique sources to find (approximate).",
        'gui_max_searches_label': "Max Searches:",
        'gui_max_searches_tooltip': "Maximum number of search API calls to make.",
        'gui_study_mode_label': "Study Mode:",
        'gui_study_mode_tooltip': "Testing processes only 2 sections; Full Study processes all outline sections.",
        'gui_output_path_label': "Output Path:",
        'gui_output_path_tooltip': "Directory where JSON data and TXT report will be saved.",
        'gui_browse_button': "Browse...",
        'gui_mode_testing': "Testing (Fast)",
        'gui_mode_full': "Full Study",
        'gui_max_tokens_outline': "Max Tokens (Outline):",
        'gui_max_tokens_outline_tooltip': "Max tokens for the LLM call generating the outline.",
        'gui_max_tokens_synthesis': "Max Tokens (Section):",
        'gui_max_tokens_synthesis_tooltip': "Max tokens for the LLM call synthesizing each content section.",
        'gui_results_per_query_label': "Results/Query:",
        'gui_results_per_query_tooltip': "Number of search results to fetch from SerpApi per query.",
        'gui_max_refinements_label': "Max Query Refinements:",
        'gui_max_refinements_tooltip': "Max times to ask LLM to refine a failed search query.",
        'gui_max_tokens_query_gen_label': "Max Tokens (Query Gen):",
        'gui_max_tokens_query_gen_tooltip': "Max tokens for the LLM call generating initial search queries.",
        'gui_max_tokens_query_refine_label': "Max Tokens (Query Refine):",
        'gui_max_tokens_query_refine_tooltip': "Max tokens for the LLM call refining a failed search query.",
    },
    'zh': {
        'system_role': "You are an advanced research assistant responding in Chinese (简体中文).",
        'plan_instruction': "Please respond concisely in English.",
        'info_instruction': "请用简体中文简洁地提供信息。",
        'outline_instruction': "请使用标准中文大纲格式（例如：一、 (一) 1.）。请用简体中文回答。",
        'synthesis_instruction': "请*仅*基于本节提供的源材料，用简体中文简洁地撰写内容。",
        'citation_instruction_zh': """
            7. **引用说明:** 请使用背景信息中提供的来源编号对应的内联引文（例如 [来源 1], [来源 5]）来引用事实。准确性至关重要。
            """,
        'citation_instruction_no_source_zh': """
            7. **引用说明:** 本节未提供具体的来源材料，如有必要请依赖通用知识，但需说明来源材料缺失。请勿添加引文。
            """,
        'not_generated': "未生成。",
        'content_warning': "(内容基于通过SerpApi收集的来源 - 通用网络搜索)",
        'char_count_label': "字符数",
        'char_count_total_label': "累计估算字符数",
        'final_char_count_label': "最终估算字符数",
        'section_not_generated': "\n*此部分内容未生成。*\n",
        'search_fail_text': "SerpApi搜索失败或未返回结果。",
        'references_heading': "收集的来源 (通过 SerpApi)",
        'no_references': "[未收集到来源或SerpApi密钥丢失/无效]",
        # GUI Labels & Tooltips
        'gui_topic_label': "研究主题:",
        'gui_topic_tooltip': "输入主要研究主题或问题。",
        'gui_language_label': "输出语言:",
        'gui_language_tooltip': "选择最终报告大纲和内容的语言。",
        'gui_start_button': "开始研究",
        'gui_status_ready': "就绪。请输入主题和语言。",
        'gui_status_running': "研究进行中...",
        'gui_status_done': "研究过程结束。",
        'gui_status_error': "发生错误。",
        'gui_keys_missing_title': "API密钥缺失",
        'gui_keys_missing_msg': "错误：未设置 OPENROUTER_API_KEY 环境变量。\n请在运行前设置它（以及用于网络搜索的 SERPAPI_API_KEY）。",
        'gui_serpapi_missing_warning': "警告：未提供 SerpApi 密钥。将跳过网络搜索步骤。",
        'gui_target_sources_label': "目标来源数:",
        'gui_target_sources_tooltip': "期望找到的唯一来源的大致数量。",
        'gui_max_searches_label': "最大搜索次数:",
        'gui_max_searches_tooltip': "允许进行的最大搜索 API 调用次数。",
        'gui_study_mode_label': "研究模式:",
        'gui_study_mode_tooltip': "测试模式仅处理2个章节；完整研究处理所有大纲章节。",
        'gui_output_path_label': "输出路径:",
        'gui_output_path_tooltip': "用于保存JSON数据和TXT报告的目录。",
        'gui_browse_button': "浏览...",
        'gui_mode_testing': "测试 (快速)",
        'gui_mode_full': "完整研究",
        'gui_max_tokens_outline': "最大令牌数(大纲):",
        'gui_max_tokens_outline_tooltip': "生成大纲的LLM调用所允许的最大令牌数。",
        'gui_max_tokens_synthesis': "最大令牌数(章节):",
        'gui_max_tokens_synthesis_tooltip': "综合每个内容章节的LLM调用所允许的最大令牌数。",
        'gui_results_per_query_label': "每次查询结果数:",
        'gui_results_per_query_tooltip': "每次查询从SerpApi获取的搜索结果数量。",
        'gui_max_refinements_label': "最大查询优化次数:",
        'gui_max_refinements_tooltip': "允许LLM优化失败搜索查询的最大次数。",
        'gui_max_tokens_query_gen_label': "最大令牌数(查询生成):",
        'gui_max_tokens_query_gen_tooltip': "生成初始搜索查询的LLM调用所允许的最大令牌数。",
        'gui_max_tokens_query_refine_label': "最大令牌数(查询优化):",
        'gui_max_tokens_query_refine_tooltip': "优化失败搜索查询的LLM调用所允许的最大令牌数。",
    }
}


# --- Core API Interaction Function (for LLM) ---
def call_llm_api(prompt, lang_code='en', model=DEEPSEEK_MODEL, max_tokens=2000, temperature=0.7):
    """Sends a prompt to the configured LLM API and returns the response."""
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured."
    system_message_base = LANG_STRINGS[lang_code]['system_role']
    system_message = f"{system_message_base} Your primary goals are:\n1. Accuracy & Detail\n2. Instruction Adherence (including language: {lang_code})\n3. Consistency\n4. Clarity\n5. Conditional Citation (follow instructions in user prompt)"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    print(f"\n--- Sending Prompt to LLM {model} via OpenRouter (Lang: {lang_code}) ---")
    print(f"Prompt (first 300 chars): {prompt[:300]}...")
    print(f"(Requesting max_tokens: {max_tokens})")
    response = None
    try:
        print(f"--- Making requests.post call to {OPENROUTER_API_URL} (Timeout={180}s) ---")
        current_llm_headers = LLM_HEADERS.copy()
        current_llm_headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        response = requests.post(OPENROUTER_API_URL, headers=current_llm_headers, json=payload, timeout=180)
        print(f"--- requests.post call returned (Status: {response.status_code}) ---")
        if response.status_code != 200:
            print(f"Error Response Text: {response.text}")
            response.raise_for_status()
        result = response.json()
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
    except requests.exceptions.Timeout:
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed ({e})"
    except json.JSONDecodeError:
        return "Error: Invalid JSON response"
    except Exception as e:
        return f"Error: An unexpected error occurred ({e})."


# --- Helper Function (Implemented External Search using SerpApi) ---
def perform_external_search(query: str, num_results: int = DEFAULT_SEARCH_RESULTS_PER_QUERY):
    """
    Performs a general web search using the SerpApi Google Search API.
    Uses hl=en for searching. Returns {'text': ..., 'source_url': ...} or None.
    """
    if not SERPAPI_API_KEY:
        print("--- SerpApi API Key not provided. Skipping web search. ---")
        return None
    hl_param = 'en'
    params = {"q": query, "api_key": SERPAPI_API_KEY, "num": num_results, "hl": hl_param, "engine": "google"}
    print(f"--- Calling SerpApi Search ---")
    print(f"--- Query: {query} ---")
    print(f"--- Num Results: {num_results}, Lang (hl): {hl_param} ---")
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
            if "error" in results:
                print(f"--- SerpApi returned an error message: {results['error']} ---")
            return None
    except requests.exceptions.Timeout:
        print("--- Error: SerpApi request timed out. ---")
        return None
    except requests.exceptions.RequestException as e:
        print(f"--- Error: SerpApi request failed: {e} ---")
        return None
    except json.JSONDecodeError:
        print("--- Error: Failed to decode JSON response from SerpApi. ---")
        return None
    except Exception as e:
        print(f"--- An unexpected error occurred during SerpApi search: {e} ---")
        return None


# --- Helper Function (Placeholder for PDF Text Extraction) ---
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
def run_research_process(topic, lang_code='en', output_queue=None,
                         # Parameters passed from GUI
                         target_model_id=DEEPSEEK_MODEL,
                         target_sources=DEFAULT_TARGET_SOURCE_COUNT,
                         max_searches=DEFAULT_MAX_SEARCH_ATTEMPTS,
                         study_mode='testing',
                         max_refinements=DEFAULT_MAX_QUERY_REFINEMENT_ATTEMPTS,
                         max_tokens_outline=DEFAULT_MAX_TOKENS_OUTLINE,
                         max_tokens_synthesis=DEFAULT_MAX_TOKENS_SYNTHESIS_PER_SECTION,
                         search_results_per_q=DEFAULT_SEARCH_RESULTS_PER_QUERY,
                         # New token limits from GUI
                         max_tokens_query_gen=DEFAULT_MAX_TOKENS_QUERY_GENERATION,
                         max_tokens_query_refine=DEFAULT_MAX_TOKENS_QUERY_REFINEMENT):
    """
    Main research process function, adapted for GUI. Runs the source-first workflow.
    Uses parameters passed from the GUI, including study_mode and token limits.
    """
    def log_message(message):
        if output_queue:
            output_queue.put(str(message))
        else:
            print(message)

    log_message(f"\n=== Starting Research Process for: {topic} (Language: {lang_code}, Mode: {study_mode}) ===")
    lang_strings = LANG_STRINGS[lang_code]
    research_data = {"topic": topic, "sources": [], "outline": "", "synthesis": {}, "estimated_char_count": 0}
    log_message(f"--- Using LLM Model ID: {target_model_id} ---")

    # --- Step 1: Source Gathering ---
    log_message(f"\n[Step 1: Gathering Sources via SerpApi (Target: {target_sources}, Max Attempts: {max_searches}, Results/Query: {search_results_per_q}, Max Refinements: {max_refinements})]")
    log_message("--- Generating OPTIMIZED search queries using LLM ---")
    query_gen_prompt = f"""
    Based on the research topic "{topic}", generate a list of 5 diverse and optimized English Google search queries... (prompt remains same)
    Output ONLY the list of queries, one query per line.
    """
    llm_generated_queries_str = call_llm_api(query_gen_prompt, lang_code='en', model=target_model_id, max_tokens=max_tokens_query_gen)
    search_queries = []
    if not llm_generated_queries_str.startswith("Error:") and llm_generated_queries_str:
        search_queries = [q.strip() for q in llm_generated_queries_str.splitlines() if q.strip()]
        search_queries = [q for q in search_queries if len(q) > 3 and not q.startswith("Here") and not q.startswith("1.")]
        log_message(f"--- LLM generated {len(search_queries)} initial search queries: ---")
        for q in search_queries:
            log_message(f"  - {q}")
    else:
        log_message("--- Warning: Failed to generate search queries via LLM. Using basic topic query. ---")
        search_queries = []
    if not search_queries:
        log_message("--- Using fallback basic queries ---")
        search_queries = [f'"{topic}"', f'"{topic}" review', f'"{topic}" survey']

    gathered_sources_dict = {}
    query_attempts = 0
    refinement_attempts = 0
    max_query_attempts = max_searches
    max_query_refinements = max_refinements
    query_index = 0
    while len(gathered_sources_dict) < target_sources and query_attempts < max_query_attempts:
        if not search_queries:
            log_message("--- ERROR: No valid search queries available. Stopping search. ---")
            break
        current_query = search_queries[query_index % len(search_queries)]
        query_attempts += 1
        log_message(f"\n--- Search Attempt {query_attempts}/{max_query_attempts}: Query: '{current_query}' ---")
        search_result = perform_external_search(current_query, num_results=search_results_per_q)
        search_succeeded = False
        if search_result and search_result.get('source_url'):
            source_url = search_result['source_url']
            if "google.com/search" not in source_url and "bing.com/search" not in source_url:
                if source_url not in gathered_sources_dict:
                    gathered_text = search_result.get('text', f"Snippet unavailable for {source_url}")
                    gathered_sources_dict[source_url] = {'text': gathered_text, 'source_url': source_url}
                    log_message(f"--- Collected source {len(gathered_sources_dict)}/{target_sources}: {source_url} ---")
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

        if not search_succeeded and refinement_attempts < max_query_refinements and query_attempts < max_query_attempts:
            refinement_attempts += 1
            log_message(f"--- Search failed. Attempting LLM query refinement ({refinement_attempts}/{max_query_refinements}) ---")
            refinement_prompt = f"""
            The previous English Google search query '{current_query}' for the overall topic '{topic}' failed to return useful results.
            Please generate one single, alternative English search query that is substantially different and might yield better results. Output only the new query string.
            """
            new_query_str = call_llm_api(refinement_prompt, lang_code='en', model=target_model_id, max_tokens=max_tokens_query_refine)
            if not new_query_str.startswith("Error:") and len(new_query_str) > 3:
                new_query_str = new_query_str.strip('\'" \n\t')
                log_message(f"--- LLM suggested new query: '{new_query_str}' ---")
                search_queries[query_index % len(search_queries)] = new_query_str
            else:
                log_message("--- LLM failed to generate refinement query. Continuing with next original query. ---")
                query_index += 1
        else:
            query_index += 1
        if len(gathered_sources_dict) >= target_sources:
            log_message(f"--- Reached target source count ({target_sources}). Stopping search. ---")
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
    log_message(f"\n[Step 2: Generating Outline from Sources ({lang_code})]")
    outline_prompt = f"""
    Based on the following {len(research_data['sources'])} collected source materials..., generate a logical outline in {lang_code}...
    {LANG_STRINGS[lang_code]['outline_instruction']}
    Collected Source Material (Summaries):\n---\n{source_summary_for_outline[:10000]}\n---
    """
    research_data["outline"] = call_llm_api(outline_prompt, lang_code=lang_code, model=target_model_id, max_tokens=max_tokens_outline)
    if research_data["outline"].startswith("Error:"):
        log_message(f"Failed to generate outline from sources ({lang_code}). Exiting.")
        output_queue.put("===RESEARCH_ERROR===")
        return None
    log_message(f"\nGenerated Outline (Based on Sources, {lang_code}):")
    log_message(research_data["outline"])
    time.sleep(1)

    # --- Step 3: Section-by-Section Synthesis ---
    log_message(f"\n[Step 3: Synthesizing Content Section by Section ({lang_code})]")
    current_outline_sections = parse_outline_sections(research_data["outline"])
    sections_to_synthesize = []
    if study_mode == 'testing':
        sections_to_synthesize = current_outline_sections[:2]
        log_message(f"--- Running in TESTING mode. Processing only the first {len(sections_to_synthesize)} outline points for synthesis ---")
    else:  # 'full' mode
        sections_to_synthesize = current_outline_sections
        log_message(f"--- Running in FULL STUDY mode. Processing all {len(sections_to_synthesize)} outline points for synthesis ---")

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
            if research_data["sources"]:
                citation_instruction = lang_strings.get(f'citation_instruction_{lang_code}', lang_strings['citation_instruction_en'])
            else:
                citation_instruction = lang_strings.get(f'citation_instruction_no_source_{lang_code}', lang_strings['citation_instruction_no_source_en'])
            synthesis_prompt = f"""
            Objective: Write a **concise but informative** section in {lang_code}... covering: "{section_id}".
            Base your writing *primarily* on the provided source materials...
            **Full Current Outline (for context ONLY):**\n{research_data['outline']}\n---
            **Section to Write:** {section_id}\n---
            **Provided Source Materials (Use these to write the section):**\n---\n{relevant_info_prompt_block[:10000]}\n---
            Instructions: ... (rest of instructions) ...
            {citation_instruction}
            {lang_strings['synthesis_instruction']}
            """
            synthesized_content = call_llm_api(synthesis_prompt, lang_code=lang_code, model=target_model_id, max_tokens=max_tokens_synthesis)
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

    log_message(f"\n=== Research Process for: {topic} Completed (Mode: {study_mode}) ===")
    log_message(f"Final Estimated Character Count (Synthesized Content): {research_data['estimated_char_count']}")

    # --- Assemble Final Document ---
    lang_strings = LANG_STRINGS[lang_code]
    final_content_warning = "(Content based on gathered sources via SerpApi - General Search)"
    if not SERPAPI_API_KEY or not research_data.get("sources"):
        final_content_warning = "(Web search skipped or failed; content may be limited)"
    full_text = []
    full_text.append(f"# Research Document: {research_data['topic']}\n")
    full_text.append(f"\n## Outline (Generated from Sources) ({lang_code.upper()})")
    full_text.append(research_data.get('outline', lang_strings['not_generated']))
    full_text.append(f"\n## Synthesized Content {final_content_warning}")
    processed_sections = list(research_data['synthesis'].keys())
    final_outline_sections = parse_outline_sections(research_data['outline'])
    ordered_sections_to_print = [s for s in final_outline_sections if s in processed_sections]
    for section_key in processed_sections:
        if section_key not in ordered_sections_to_print:
            ordered_sections_to_print.append(section_key)
    if ordered_sections_to_print:
        for section in ordered_sections_to_print:
            content = research_data['synthesis'].get(section, lang_strings['section_not_generated'])
            full_text.append(f"\n### {section.strip()}")
            full_text.append(content)
        sections_processed_count = len(ordered_sections_to_print)
        if study_mode == 'testing' and sections_processed_count < len(final_outline_sections):
            full_text.append(f"\n\n[Note: Only the first {sections_processed_count} sections were processed in Testing Mode.]")
        elif study_mode == 'full' and sections_processed_count < len(final_outline_sections):
            full_text.append(f"\n\n[Note: Processed {sections_processed_count}/{len(final_outline_sections)} sections in Full Study Mode.]")
    else:
        full_text.append("\n[No sections were synthesized in this run.]")
    full_text.append(f"\n\n## {lang_strings['references_heading']}\n")
    if research_data.get("sources"):
        sorted_sources = sorted(research_data["sources"], key=lambda x: x.get('source_url', ''))
        for i, src in enumerate(sorted_sources):
            url = src.get('source_url', 'N/A')
            text_snippet = src.get('text', '')[:120]
            full_text.append(f"{i+1}. URL: {url}\n   Info: {text_snippet}...")
    else:
        full_text.append(lang_strings['no_references'])
    assembled_document = "\n".join(full_text)

    # Put final result and data in queue for GUI
    if output_queue:
        output_queue.put("\n\n=== Final Assembled Document ===\n")
        output_queue.put(assembled_document)
        output_queue.put({"type": "save_data", "data": research_data, "assembled_doc": assembled_document})
    else:
        print("\n\n=== Final Assembled Document ===\n")
        print(assembled_document)

    return research_data  # Return data


# --- Tooltip Helper Class ---
class Tooltip:
    """Creates a tooltip (pop-up window) for a given widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, justify='left', background="#ffffe0",
                         relief='solid', borderwidth=1, font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = None


# --- GUI Application Class ---
class ResearchApp:
    def __init__(self, root):
        """Initializes the Tkinter GUI application."""
        self.root = root
        self.root.title("DSDeepResearch_Lite GUI v1.10")  # Version bump
        self.root.minsize(650, 800)  # Adjusted size
        self.style = ttk.Style(self.root)
        try:
            if "clam" in self.style.theme_names():
                self.style.theme_use('clam')
            elif "vista" in self.style.theme_names():
                self.style.theme_use('vista')
            elif "aqua" in self.style.theme_names():
                self.style.theme_use('aqua')
        except tk.TclError:
            print("Default theme not found, using fallback.")

        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=10)
        text_font = font.nametofont("TkTextFont")
        text_font.configure(size=10)
        self.root.option_add("*Font", default_font)

        # --- Tkinter Variables ---
        self.topic_var = tk.StringVar()
        self.lang_var = tk.StringVar(value='en')
        self.status_var = tk.StringVar(value="Initializing...")
        self.target_sources_var = tk.IntVar(value=DEFAULT_TARGET_SOURCE_COUNT)
        self.max_searches_var = tk.IntVar(value=DEFAULT_MAX_SEARCH_ATTEMPTS)
        self.study_mode_var = tk.StringVar(value='testing')
        self.output_path_var = tk.StringVar(value=DEFAULT_OUTPUT_PATH)
        self.max_tokens_outline_var = tk.IntVar(value=DEFAULT_MAX_TOKENS_OUTLINE)
        self.max_tokens_synthesis_var = tk.IntVar(value=DEFAULT_MAX_TOKENS_SYNTHESIS_PER_SECTION)
        self.results_per_query_var = tk.IntVar(value=DEFAULT_SEARCH_RESULTS_PER_QUERY)
        self.max_refinements_var = tk.IntVar(value=DEFAULT_MAX_QUERY_REFINEMENT_ATTEMPTS)
        # New variables for Query Gen/Refine tokens
        self.max_tokens_query_gen_var = tk.IntVar(value=DEFAULT_MAX_TOKENS_QUERY_GENERATION)
        self.max_tokens_query_refine_var = tk.IntVar(value=DEFAULT_MAX_TOKENS_QUERY_REFINEMENT)

        self.output_queue = queue.Queue()
        self.final_data_for_saving = None

        self.create_widgets()
        self.check_api_keys()

    def select_output_path(self):
        """Opens a directory chooser dialog and updates the output path variable."""
        directory = filedialog.askdirectory(initialdir=self.output_path_var.get())
        if directory:
            self.output_path_var.set(directory)

    def create_widgets(self):
        """Creates and lays out the GUI widgets."""
        ui_lang = self.lang_var.get()
        ui_strings = LANG_STRINGS[ui_lang]

        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(main_frame, text="Input & Parameters", padding="10")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1, minsize=150)
        input_frame.columnconfigure(3, weight=1, minsize=150)

        # Row 0: Topic
        topic_label = ttk.Label(input_frame, text=ui_strings['gui_topic_label'])
        topic_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        Tooltip(topic_label, ui_strings['gui_topic_tooltip'])
        self.topic_entry = ttk.Entry(input_frame, textvariable=self.topic_var, width=60)
        self.topic_entry.grid(row=0, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=3)

        # Row 1: Language & Study Mode
        lang_label = ttk.Label(input_frame, text=ui_strings['gui_language_label'])
        lang_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        Tooltip(lang_label, ui_strings['gui_language_tooltip'])
        lang_frame = ttk.Frame(input_frame)
        lang_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)
        en_radio = ttk.Radiobutton(lang_frame, text="English", variable=self.lang_var, value='en')
        en_radio.pack(side=tk.LEFT, padx=0)
        zh_radio = ttk.Radiobutton(lang_frame, text="Chinese (简体)", variable=self.lang_var, value='zh')
        zh_radio.pack(side=tk.LEFT, padx=5)

        mode_label = ttk.Label(input_frame, text=ui_strings['gui_study_mode_label'])
        mode_label.grid(row=1, column=2, sticky=tk.W, padx=(20, 5), pady=3)
        Tooltip(mode_label, ui_strings['gui_study_mode_tooltip'])
        mode_frame = ttk.Frame(input_frame)
        mode_frame.grid(row=1, column=3, sticky=tk.W, padx=5, pady=3)
        testing_radio = ttk.Radiobutton(mode_frame, text=ui_strings['gui_mode_testing'], variable=self.study_mode_var, value='testing')
        testing_radio.pack(side=tk.LEFT, padx=0)
        full_radio = ttk.Radiobutton(mode_frame, text=ui_strings['gui_mode_full'], variable=self.study_mode_var, value='full')
        full_radio.pack(side=tk.LEFT, padx=5)

        # Row 2: Target Sources & Max Searches
        target_sources_label = ttk.Label(input_frame, text=ui_strings['gui_target_sources_label'])
        target_sources_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        Tooltip(target_sources_label, ui_strings['gui_target_sources_tooltip'])
        self.target_sources_spinbox = ttk.Spinbox(input_frame, from_=5, to=200, increment=5, width=5, textvariable=self.target_sources_var)
        self.target_sources_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)

        max_searches_label = ttk.Label(input_frame, text=ui_strings['gui_max_searches_label'])
        max_searches_label.grid(row=2, column=2, sticky=tk.W, padx=(20, 5), pady=3)
        Tooltip(max_searches_label, ui_strings['gui_max_searches_tooltip'])
        self.max_searches_spinbox = ttk.Spinbox(input_frame, from_=1, to=200, increment=1, width=5, textvariable=self.max_searches_var)
        self.max_searches_spinbox.grid(row=2, column=3, sticky=tk.W, padx=5, pady=3)

        # Row 3: Results per Query & Max Refinements
        results_query_label = ttk.Label(input_frame, text=ui_strings['gui_results_per_query_label'])
        results_query_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        Tooltip(results_query_label, ui_strings['gui_results_per_query_tooltip'])
        self.results_query_spinbox = ttk.Spinbox(input_frame, from_=1, to=10, increment=1, width=5, textvariable=self.results_per_query_var)
        self.results_query_spinbox.grid(row=3, column=1, sticky=tk.W, padx=5, pady=3)

        max_refine_label = ttk.Label(input_frame, text=ui_strings['gui_max_refinements_label'])
        max_refine_label.grid(row=3, column=2, sticky=tk.W, padx=(20, 5), pady=3)
        Tooltip(max_refine_label, ui_strings['gui_max_refinements_tooltip'])
        self.max_refine_spinbox = ttk.Spinbox(input_frame, from_=0, to=10, increment=1, width=5, textvariable=self.max_refinements_var)
        self.max_refine_spinbox.grid(row=3, column=3, sticky=tk.W, padx=5, pady=3)

        # Row 4: Max Tokens (Outline & Synthesis)
        max_tk_outline_label = ttk.Label(input_frame, text=ui_strings['gui_max_tokens_outline'])
        max_tk_outline_label.grid(row=4, column=0, sticky=tk.W, padx=5, pady=3)
        Tooltip(max_tk_outline_label, ui_strings['gui_max_tokens_outline_tooltip'])
        self.max_tk_outline_spinbox = ttk.Spinbox(input_frame, from_=200, to=4000, increment=100, width=7, textvariable=self.max_tokens_outline_var)
        self.max_tk_outline_spinbox.grid(row=4, column=1, sticky=tk.W, padx=5, pady=3)

        max_tk_synth_label = ttk.Label(input_frame, text=ui_strings['gui_max_tokens_synthesis'])
        max_tk_synth_label.grid(row=4, column=2, sticky=tk.W, padx=(20, 5), pady=3)
        Tooltip(max_tk_synth_label, ui_strings['gui_max_tokens_synthesis_tooltip'])
        self.max_tk_synth_spinbox = ttk.Spinbox(input_frame, from_=100, to=4000, increment=100, width=7, textvariable=self.max_tokens_synthesis_var)
        self.max_tk_synth_spinbox.grid(row=4, column=3, sticky=tk.W, padx=5, pady=3)

        # Row 5: Max Tokens (Query Gen & Refine)
        max_tk_qgen_label = ttk.Label(input_frame, text=ui_strings['gui_max_tokens_query_gen_label'])
        max_tk_qgen_label.grid(row=5, column=0, sticky=tk.W, padx=5, pady=3)
        Tooltip(max_tk_qgen_label, ui_strings['gui_max_tokens_query_gen_tooltip'])
        self.max_tk_qgen_spinbox = ttk.Spinbox(input_frame, from_=50, to=1000, increment=50, width=7, textvariable=self.max_tokens_query_gen_var)
        self.max_tk_qgen_spinbox.grid(row=5, column=1, sticky=tk.W, padx=5, pady=3)

        max_tk_qref_label = ttk.Label(input_frame, text=ui_strings['gui_max_tokens_query_refine_label'])
        max_tk_qref_label.grid(row=5, column=2, sticky=tk.W, padx=(20, 5), pady=3)
        Tooltip(max_tk_qref_label, ui_strings['gui_max_tokens_query_refine_tooltip'])
        self.max_tk_qref_spinbox = ttk.Spinbox(input_frame, from_=50, to=1000, increment=50, width=7, textvariable=self.max_tokens_query_refine_var)
        self.max_tk_qref_spinbox.grid(row=5, column=3, sticky=tk.W, padx=5, pady=3)

        # Row 6: Output Path
        path_label = ttk.Label(input_frame, text=ui_strings['gui_output_path_label'])
        path_label.grid(row=6, column=0, sticky=tk.W, padx=5, pady=3)
        Tooltip(path_label, ui_strings['gui_output_path_tooltip'])
        self.path_entry = ttk.Entry(input_frame, textvariable=self.output_path_var, width=40)
        self.path_entry.grid(row=6, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=3)
        self.browse_button = ttk.Button(input_frame, text=ui_strings['gui_browse_button'], command=self.select_output_path)
        self.browse_button.grid(row=6, column=3, sticky=tk.W, padx=5, pady=3)

        # Row 7: Start Button
        self.start_button = ttk.Button(input_frame, text=ui_strings['gui_start_button'], command=self.start_research_thread)
        self.start_button.grid(row=7, column=0, columnspan=4, pady=10)

        # --- Output Frame ---
        output_frame = ttk.LabelFrame(main_frame, text="Output Log & Results", padding="10")
        output_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.rowconfigure(1, weight=1)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=20, width=80, state='disabled', font=("Courier New", 9))
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        # --- Status Bar ---
        status_bar = ttk.Frame(main_frame, relief=tk.SUNKEN, padding="2 5 2 5")
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=(5, 0))
        self.status_label = ttk.Label(status_bar, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        copyright_text = "Copyright (c) 2025 Pengwei Zhu"
        self.copyright_label = ttk.Label(status_bar, text=copyright_text, anchor=tk.E)
        self.copyright_label.pack(side=tk.RIGHT)

    def log_to_gui(self, message):
        """Appends a message to the output text area in a thread-safe way."""
        try:
            if self.output_text.winfo_exists():
                self.output_text.configure(state='normal')
                self.output_text.insert(tk.END, str(message) + "\n")
                self.output_text.configure(state='disabled')
                self.output_text.see(tk.END)
        except tk.TclError as e:
            print(f"GUI Error logging message: {e}")
        except Exception as e:
            print(f"Unexpected error in log_to_gui: {e}")

    def check_api_keys(self):
        """Checks for API keys and updates status/button."""
        lang = self.lang_var.get()
        lang_strings = LANG_STRINGS[lang]
        if not OPENROUTER_API_KEY:
            error_msg = lang_strings['gui_keys_missing_msg']
            self.status_var.set(error_msg)
            if self.root.winfo_exists():
                messagebox.showerror(lang_strings['gui_keys_missing_title'], error_msg)
            self.start_button.configure(state='disabled')
            return False
        else:
            if not SERPAPI_API_KEY:
                self.status_var.set(lang_strings['gui_serpapi_missing_warning'])
            else:
                self.status_var.set(lang_strings['gui_status_ready'])
            self.start_button.configure(state='normal')
            return True

    def start_research_thread(self):
        """Reads parameters from GUI and starts the research process in a thread."""
        if not self.check_api_keys():
            return

        topic = self.topic_var.get().strip()
        lang = self.lang_var.get()
        study_mode = self.study_mode_var.get()
        output_path = self.output_path_var.get().strip()
        lang_strings = LANG_STRINGS[lang]

        if not topic:
            messagebox.showwarning("Input Missing", "Please enter a research topic.")
            return
        if not output_path:
            messagebox.showwarning("Input Missing", "Please enter an output path.")
            return

        # --- Get Parameters from GUI with Validation ---
        try:
            target_sources = self.target_sources_var.get()
            assert target_sources >= 1
        except (tk.TclError, ValueError, AssertionError):
            messagebox.showerror("Invalid Parameter", "Target Sources must be a valid integer >= 1.")
            return
        try:
            max_searches = self.max_searches_var.get()
            assert max_searches >= 1
        except (tk.TclError, ValueError, AssertionError):
            messagebox.showerror("Invalid Parameter", "Max Searches must be a valid integer >= 1.")
            return
        try:
            max_tk_outline = self.max_tokens_outline_var.get()
            assert max_tk_outline >= 100
        except (tk.TclError, ValueError, AssertionError):
            messagebox.showerror("Invalid Parameter", "Max Tokens (Outline) must be a valid integer >= 100.")
            return
        try:
            max_tk_synthesis = self.max_tokens_synthesis_var.get()
            assert max_tk_synthesis >= 50
        except (tk.TclError, ValueError, AssertionError):
            messagebox.showerror("Invalid Parameter", "Max Tokens (Section) must be a valid integer >= 50.")
            return
        try:
            results_per_query = self.results_per_query_var.get()
            assert results_per_query >= 1
        except (tk.TclError, ValueError, AssertionError):
            messagebox.showerror("Invalid Parameter", "Results/Query must be a valid integer >= 1.")
            return
        try:
            max_refinements = self.max_refinements_var.get()
            assert max_refinements >= 0
        except (tk.TclError, ValueError, AssertionError):
            messagebox.showerror("Invalid Parameter", "Max Query Refinements must be a valid integer >= 0.")
            return
        # New token limits validation
        try:
            max_tk_query_gen = self.max_tokens_query_gen_var.get()
            assert max_tk_query_gen >= 50
        except (tk.TclError, ValueError, AssertionError):
            messagebox.showerror("Invalid Parameter", "Max Tokens (Query Gen) must be a valid integer >= 50.")
            return
        try:
            max_tk_query_refine = self.max_tokens_query_refine_var.get()
            assert max_tk_query_refine >= 20
        except (tk.TclError, ValueError, AssertionError):
            messagebox.showerror("Invalid Parameter", "Max Tokens (Query Refine) must be a valid integer >= 20.")
            return

        model_id = DEEPSEEK_MODEL  # Use fixed model ID

        # --- Start Thread ---
        self.start_button.configure(state='disabled')
        self.output_text.configure(state='normal')
        self.output_text.delete('1.0', tk.END)
        self.output_text.configure(state='disabled')
        self.status_var.set(lang_strings['gui_status_running'])
        self.log_to_gui(f"Starting research for topic: '{topic}' in language: {lang}")
        self.log_to_gui(f"Mode: {study_mode}, Target Sources: {target_sources}, Max Searches: {max_searches}, Results/Query: {results_per_query}, Max Refinements: {max_refinements}")
        self.log_to_gui(f"Max Tokens: Outline={max_tk_outline}, Section={max_tk_synthesis}, QueryGen={max_tk_query_gen}, QueryRefine={max_tk_query_refine}, Model={model_id}")
        if not SERPAPI_API_KEY:
            self.log_to_gui(lang_strings['gui_serpapi_missing_warning'])

        # Pass ALL GUI parameters to the worker function
        self.research_thread = threading.Thread(
            target=research_worker,
            args=(topic, lang, self.output_queue,
                  model_id, target_sources, max_searches, study_mode,
                  max_refinements,
                  output_path,
                  max_tk_outline, max_tk_synthesis,
                  results_per_query,
                  max_tk_query_gen, max_tk_query_refine),
            daemon=True
        )
        self.research_thread.start()
        self.root.after(100, self.process_queue)

    def save_results(self, final_data, assembled_doc, target_directory, lang_code):
        """Saves the research data and assembled document to files."""
        self.log_to_gui(f"\n--- Attempting to save results to: {target_directory} ---")
        try:
            os.makedirs(target_directory, exist_ok=True)
            topic = final_data.get("topic", "untitled")
            safe_topic = re.sub(r'[\\/*?:"<>|]', "", topic).replace(' ', '_').lower()[:50]
            base_data_filename = f"research_data_{safe_topic}_{lang_code}_source_first.json"
            base_text_filename = f"research_report_{safe_topic}_{lang_code}_source_first.txt"
            full_data_path = os.path.join(target_directory, base_data_filename)
            full_text_path = os.path.join(target_directory, base_text_filename)
            with open(full_data_path, "w", encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
            self.log_to_gui(f"Full research data saved to: {full_data_path}")
            with open(full_text_path, "w", encoding='utf-8') as f:
                f.write(assembled_doc)
            self.log_to_gui(f"Assembled document saved to: {full_text_path}")
        except Exception as e:
            error_msg = f"Error saving files: {e}"
            self.log_to_gui(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.log_to_gui(error_msg)
            self.log_to_gui(f"Path: {target_directory}")
            self.log_to_gui(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if self.root.winfo_exists():
                messagebox.showerror("File Save Error", error_msg)

    def process_queue(self):
        """Processes messages from the research thread queue to update the GUI."""
        try:
            while True:
                message = self.output_queue.get_nowait()
                lang = self.lang_var.get()
                lang_strings = LANG_STRINGS[lang]
                if isinstance(message, dict) and message.get("type") == "save_data":
                    self.final_data_for_saving = message
                elif message == "===RESEARCH_COMPLETE===":
                    self.status_var.set(lang_strings['gui_status_done'])
                    self.start_button.configure(state='normal')
                    if self.final_data_for_saving:
                        self.save_results(
                            self.final_data_for_saving['data'],
                            self.final_data_for_saving['assembled_doc'],
                            self.final_data_for_saving['path'],
                            self.final_data_for_saving['lang_code']
                        )
                        self.final_data_for_saving = None
                elif message == "===RESEARCH_ERROR===":
                    self.status_var.set(lang_strings['gui_status_error'])
                    self.start_button.configure(state='normal')
                    self.final_data_for_saving = None
                else:
                    self.log_to_gui(message)
        except queue.Empty:
            pass

        if hasattr(self, 'research_thread') and self.research_thread.is_alive():
            self.root.after(200, self.process_queue)
        elif hasattr(self, 'research_thread') and not self.research_thread.is_alive():
            self.final_data_for_saving = None
            if self.start_button['state'] == 'disabled':
                self.start_button.configure(state='normal')
                current_status = self.status_var.get()
                lang = self.lang_var.get()
                lang_strings = LANG_STRINGS[lang]
                if lang_strings['gui_status_done'] not in current_status and lang_strings['gui_status_error'] not in current_status:
                    self.status_var.set(lang_strings['gui_status_done'] + " (Thread ended)")

# --- Worker Function (Runs in Background Thread) ---
def research_worker(topic, lang_code, output_queue,
                    # Accept parameters from GUI
                    model_id, target_sources, max_searches, study_mode,
                    max_refinements, output_path,
                    max_tk_outline, max_tk_synthesis,
                    results_per_q,
                    max_tk_query_gen, max_tk_query_refine):
    """
    Wrapper function to run the research process and put output/status in the queue.
    This function runs in the background thread. Redirects print to queue.
    """
    class QueueLogger:
        def __init__(self, queue):
            self.queue = queue

        def write(self, message):
            if message and not message.isspace():
                self.queue.put(message.rstrip())

        def flush(self):
            pass

    original_stdout = sys.stdout
    sys.stdout = QueueLogger(output_queue)
    final_data = None
    assembled_document = "[Document not assembled]"
    try:
        final_data = run_research_process(
            topic=topic,
            lang_code=lang_code,
            output_queue=output_queue,
            target_model_id=model_id,
            target_sources=target_sources,
            max_searches=max_searches,
            study_mode=study_mode,
            max_refinements=max_refinements,
            max_tokens_outline=max_tk_outline,
            max_tokens_synthesis=max_tk_synthesis,
            search_results_per_q=results_per_q,
            max_tokens_query_gen=max_tk_query_gen,
            max_tokens_query_refine=max_tk_query_refine
        )

        if final_data:
            lang_strings = LANG_STRINGS[lang_code]
            final_content_warning = "(Content based on gathered sources via SerpApi - General Search)"
            if not SERPAPI_API_KEY or not final_data.get("sources"):
                final_content_warning = "(Web search skipped or failed; content may be limited)"
            full_text = []
            full_text.append(f"# Research Document: {final_data['topic']}\n")
            full_text.append(f"\n## Outline (Generated from Sources) ({lang_code.upper()})")
            full_text.append(final_data.get('outline', lang_strings['not_generated']))
            full_text.append(f"\n## Synthesized Content {final_content_warning}")
            processed_sections = list(final_data['synthesis'].keys())
            final_outline_sections = parse_outline_sections(final_data['outline'])
            ordered_sections_to_print = [s for s in final_outline_sections if s in processed_sections]
            for section_key in processed_sections:
                if section_key not in ordered_sections_to_print:
                    ordered_sections_to_print.append(section_key)
            if ordered_sections_to_print:
                for section in ordered_sections_to_print:
                    content = final_data['synthesis'].get(section, lang_strings['section_not_generated'])
                    full_text.append(f"\n### {section.strip()}")
                    full_text.append(content)
                sections_processed_count = len(ordered_sections_to_print)
                if study_mode == 'testing' and sections_processed_count < len(final_outline_sections):
                    full_text.append(f"\n\n[Note: Only the first {sections_processed_count} sections were processed in Testing Mode.]")
                elif study_mode == 'full' and sections_processed_count < len(final_outline_sections):
                    full_text.append(f"\n\n[Note: Processed {sections_processed_count}/{len(final_outline_sections)} sections in Full Study Mode.]")
            else:
                full_text.append("\n[No sections were synthesized in this run.]")
            full_text.append(f"\n\n## {lang_strings['references_heading']}\n")
            if final_data.get("sources"):
                sorted_sources = sorted(final_data["sources"], key=lambda x: x.get('source_url', ''))
                for i, src in enumerate(sorted_sources):
                    url = src.get('source_url', 'N/A')
                    text_snippet = src.get('text', '')[:120]
                    full_text.append(f"{i+1}. URL: {url}\n   Info: {text_snippet}...")
            else:
                full_text.append(lang_strings['no_references'])
            assembled_document = "\n".join(full_text)

            output_queue.put({
                "type": "save_data",
                "data": final_data,
                "assembled_doc": assembled_document,
                "path": output_path,
                "lang_code": lang_code
            })
            output_queue.put("===RESEARCH_COMPLETE===")
        else:
            output_queue.put("Research process returned no data (likely due to an earlier error).")
            output_queue.put("===RESEARCH_ERROR===")

    except Exception as e:
        output_queue.put(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        output_queue.put(f"An error occurred in the research worker thread: {e}")
        output_queue.put(traceback.format_exc())
        output_queue.put(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        output_queue.put("===RESEARCH_ERROR===")
    finally:
        sys.stdout = original_stdout

# --- Main Execution ---
if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("CRITICAL ERROR: OPENROUTER_API_KEY environment variable not set.")
        print("Please set the environment variable and restart.")
        try:
            root_err = tk.Tk()
            root_err.withdraw()
            messagebox.showerror("API Key Missing", "CRITICAL ERROR: OPENROUTER_API_KEY environment variable not set.\nPlease set it and restart the application.")
            root_err.destroy()
        except Exception as tk_err:
            print(f"Tkinter error box failed: {tk_err}")
        exit()

    root = tk.Tk()
    app = ResearchApp(root)
    root.mainloop()  # Start the Tkinter event loop
