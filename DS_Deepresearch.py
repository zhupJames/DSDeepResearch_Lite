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
import sys # Added for exit
import traceback # For detailed error logging
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

# --- Load API Keys ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# *** Check for essential keys on script start ***
if not OPENROUTER_API_KEY:
    print("CRITICAL ERROR: OPENROUTER_API_KEY environment variable not set.")
    print("Please set the environment variable and restart.")
    sys.exit(1) # Exit if essential key is missing
else:
    print("OpenRouter API Key found.")

if not SERPAPI_API_KEY:
    print("Startup Warning: SERPAPI_API_KEY not set. Web search step will be skipped.")
else:
    print("SerpApi API Key found.")


# *** API and Model Configuration ***
DEEPSEEK_MODEL = "deepseek/deepseek-chat" # Fixed LLM Model ID
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_ENDPOINT = "https://serpapi.com/search" # SerpApi endpoint

# Headers for OpenRouter LLM calls
LLM_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "YOUR_APP_URL_OR_NAME", # Optional: Replace
    "X-Title": "DSDeepResearch_Lite CLI" # Optional: Replace
}

# --- Default Constants (used if user doesn't provide input) ---
# Token limits for API calls
DEFAULT_MAX_TOKENS_SYNTHESIS_PER_SECTION = 500
DEFAULT_MAX_TOKENS_OUTLINE = 1500
DEFAULT_MAX_TOKENS_QUERY_GENERATION = 300
DEFAULT_MAX_TOKENS_QUERY_REFINEMENT = 100

# Source Gathering Parameters
DEFAULT_TARGET_SOURCE_COUNT = 40
DEFAULT_SEARCH_RESULTS_PER_QUERY = 5
DEFAULT_MAX_SEARCH_ATTEMPTS = 10
DEFAULT_MAX_QUERY_REFINEMENT_ATTEMPTS = 3

# Default output path
DEFAULT_OUTPUT_PATH = os.path.join(os.getcwd(), "Deepresearch_Output") # Default to subfolder

# --- Language Strings (Simplified for backend/CLI focus) ---
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
        'final_char_count_label': "Final Estimated Word Count",
        'section_not_generated': "\n*Content not generated for this section.*\n",
        'search_fail_text': "SerpApi search failed or returned no results.",
        'references_heading': "Collected Sources (via SerpApi)",
        'no_references': "[No sources collected or SerpApi key missing/invalid]",
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
        'final_char_count_label': "最终估算字符数",
        'section_not_generated': "\n*此部分内容未生成。*\n",
        'search_fail_text': "SerpApi搜索失败或未返回结果。",
        'references_heading': "收集的来源 (通过 SerpApi)",
        'no_references': "[未收集到来源或SerpApi密钥丢失/无效]",
    }
}


# --- Core API Interaction Function (for LLM) ---
def call_llm_api(prompt, lang_code='en', model=DEEPSEEK_MODEL, max_tokens=2000, temperature=0.7):
    """Sends a prompt to the configured LLM API and returns the response content."""
    # (Function body remains the same)
    if not OPENROUTER_API_KEY: return "Error: OPENROUTER_API_KEY not configured."
    system_message_base = LANG_STRINGS[lang_code]['system_role']
    system_message = f"{system_message_base} Your primary goals are:\n1. Accuracy & Detail\n2. Instruction Adherence (including language: {lang_code})\n3. Consistency\n4. Clarity\n5. Conditional Citation (follow instructions in user prompt)"
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
    except requests.exceptions.Timeout: return "Error: API request timed out."
    except requests.exceptions.RequestException as e: return f"Error: API request failed ({e})"
    except json.JSONDecodeError: return "Error: Invalid JSON response"
    except Exception as e: return f"Error: An unexpected error occurred ({e})."


# --- Helper Function (Implemented External Search using SerpApi) ---
def perform_external_search(query: str, num_results: int = DEFAULT_SEARCH_RESULTS_PER_QUERY):
    """
    Performs a general web search using the SerpApi Google Search API.
    Uses hl=en for searching. Returns {'text': ..., 'source_url': ...} or None.
    """
    # (Function body remains the same)
    if not SERPAPI_API_KEY:
        print("--- SerpApi API Key not provided. Skipping web search. ---")
        return None
    hl_param = 'en'
    params = { "q": query, "api_key": SERPAPI_API_KEY, "num": num_results, "hl": hl_param, "engine": "google" }
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
            if "error" in results: print(f"--- SerpApi returned an error message: {results['error']} ---")
            return None
    except requests.exceptions.Timeout: print("--- Error: SerpApi request timed out. ---"); return None
    except requests.exceptions.RequestException as e: print(f"--- Error: SerpApi request failed: {e} ---"); return None
    except json.JSONDecodeError: print("--- Error: Failed to decode JSON response from SerpApi. ---"); return None
    except Exception as e: print(f"--- An unexpected error occurred during SerpApi search: {e} ---"); return None


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
    # (Function body remains the same)
    pattern = r"^\s*(?:[IVXLCDM]+\.|[A-Z]\.|[0-9]+\.|[a-z]\.|[一二三四五六七八九十]+[、．\.])\s+.*"
    sections = []
    if not isinstance(outline_text, str): return sections
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
         print("Warning: Basic outline parsing failed. Using non-empty lines as sections.")
    print(f"Parsed {len(sections)} potential sections from outline using basic regex.")
    return sections


# --- Research Workflow Simulation (Source-First with SerpApi, General Search) ---
def run_research_process(topic, lang_code='en',
                         # Parameters passed from main execution block
                         target_model_id=DEEPSEEK_MODEL,
                         target_sources=DEFAULT_TARGET_SOURCE_COUNT,
                         max_searches=DEFAULT_MAX_SEARCH_ATTEMPTS,
                         study_mode='testing',
                         max_refinements=DEFAULT_MAX_QUERY_REFINEMENT_ATTEMPTS,
                         max_tokens_outline=DEFAULT_MAX_TOKENS_OUTLINE,
                         max_tokens_synthesis=DEFAULT_MAX_TOKENS_SYNTHESIS_PER_SECTION,
                         search_results_per_q=DEFAULT_SEARCH_RESULTS_PER_QUERY,
                         max_tokens_query_gen=DEFAULT_MAX_TOKENS_QUERY_GENERATION,
                         max_tokens_query_refine=DEFAULT_MAX_TOKENS_QUERY_REFINEMENT):
    """
    Main research process function for command-line execution.

    Args:
        topic (str): The research topic.
        lang_code (str): The target language code ('en' or 'zh').
        target_model_id (str): The LLM model ID to use.
        target_sources (int): The desired number of sources to collect.
        max_searches (int): The maximum number of SerpApi calls.
        study_mode (str): 'testing' (process 2 sections) or 'full' (process all).
        max_refinements (int): Max times to ask LLM to refine failed queries.
        max_tokens_outline (int): Max tokens for outline generation LLM call.
        max_tokens_synthesis (int): Max tokens for each section synthesis LLM call.
        search_results_per_q (int): Number of results to fetch per search query.
        max_tokens_query_gen (int): Max tokens for initial query generation call.
        max_tokens_query_refine (int): Max tokens for query refinement call.

    Returns:
        dict or None: The research_data dictionary containing results, or None on critical failure.
    """
    # Use standard print for logging in CLI version
    log_message = print

    log_message(f"\n=== Starting Research Process for: {topic} (Language: {lang_code}, Mode: {study_mode}) ===")
    lang_strings = LANG_STRINGS[lang_code]
    research_data = { "topic": topic, "sources": [], "outline": "", "synthesis": {} } # Removed char count here
    log_message(f"--- Using LLM Model ID: {target_model_id} ---")

    try:
        # --- Step 1: Source Gathering ---
        log_message(f"\n[Step 1: Gathering Sources via SerpApi (Target: {target_sources}, Max Attempts: {max_searches}, Results/Query: {search_results_per_q}, Max Refinements: {max_refinements})]")
        log_message("--- Generating OPTIMIZED search queries using LLM ---")
        query_gen_prompt = f"""
        Based on the research topic "{topic}", generate a list of 5 diverse and optimized English Google search queries designed to find the most relevant and high-quality sources (like academic papers, reputable web resources).
        Focus on precision, key technical terms, synonyms, and potential variations that capture the core concepts effectively for maximizing relevant search results.
        Output ONLY the list of queries, one query per line. Do not add numbering or introductory text.
        """
        llm_generated_queries_str = call_llm_api(query_gen_prompt, lang_code='en', model=target_model_id, max_tokens=max_tokens_query_gen)
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
                if search_result is None and SERPAPI_API_KEY: log_message("--- Search failed. Check SerpApi key and service status. ---")
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
            return None # Indicate failure

        source_summary_for_outline = ""
        for i, src in enumerate(research_data["sources"]):
            source_summary_for_outline += f"Source {i+1} ({src.get('source_url', 'N/A')}):\n{src.get('text', 'N/A')[:500]}...\n\n"

        # --- Step 2: Outline Generation ---
        log_message(f"\n[Step 2: Generating Outline from Sources ({lang_code})]...")
        outline_prompt = f"""
        Based on the following {len(research_data['sources'])} collected source materials..., generate a logical outline in {lang_code}...
        {LANG_STRINGS[lang_code]['outline_instruction']}
        Collected Source Material (Summaries):\n---\n{source_summary_for_outline[:10000]}\n---
        """
        research_data["outline"] = call_llm_api(outline_prompt, lang_code=lang_code, model=target_model_id, max_tokens=max_tokens_outline)
        if research_data["outline"].startswith("Error:"):
            log_message(f"Failed to generate outline from sources ({lang_code}). Exiting.")
            return None
        log_message(f"\nGenerated Outline (Based on Sources, {lang_code}):")
        log_message(research_data["outline"])
        time.sleep(1)

        # --- Step 3: Section-by-Section Synthesis ---
        log_message(f"\n[Step 3: Synthesizing Content Section by Section ({lang_code})]...")
        current_outline_sections = parse_outline_sections(research_data["outline"])
        sections_to_synthesize = []
        if study_mode == 'testing':
            sections_to_synthesize = current_outline_sections[:2]
            log_message(f"--- Running in TESTING mode. Processing only the first {len(sections_to_synthesize)} outline points for synthesis ---")
        else: # 'full' mode
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
                     source_num = idx + 1; url = src.get('source_url', 'N/A'); text_snippet = src.get('text', 'N/A')[:1500]
                     relevant_info_parts.append(f"Source {source_num}:\n{text_snippet}\n(URL: {url})")
                     source_mapping[source_num] = url
                relevant_info_prompt_block = "\n\n".join(relevant_info_parts)
                if research_data["sources"]: citation_instruction = lang_strings[f'citation_instruction_{lang_code}']
                else: citation_instruction = lang_strings[f'citation_instruction_no_source_{lang_code}']
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
                    # log_message(f"\nSynthesized Content for '{section_id}' ({lang_strings['char_count_label']}: {section_char_count}):")
                    # log_message(synthesized_content[:300] + "...") # Reduce console noise
                else:
                    log_message(f"Failed to synthesize content for section '{section_id}'.")
                    research_data["synthesis"][section_id] = f"Error: Failed to generate content for {section_id}."
                # research_data["estimated_char_count"] = total_synthesized_chars # Not needed here
                # log_message(f"{lang_strings['char_count_total_label']}: {total_synthesized_chars}") # Reduce noise
                time.sleep(1) # Shorter delay for CLI

        log_message(f"--- Synthesis Complete for {len(sections_to_synthesize)} sections. ---")

        log_message(f"\n=== Research Process Completed (Mode: {study_mode}) ===")
        return research_data # Return all collected data

    except Exception as e:
        # Log any unexpected error during the process
        log_message(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        log_message(f"An error occurred during the research process: {e}")
        log_message(traceback.format_exc())
        log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None # Indicate failure


# --- Function to get integer input with default ---
def get_int_input(prompt_text, default_value):
    """Gets integer input from user, returns default if input is invalid/empty."""
    while True:
        user_input = input(f"{prompt_text} (Default: {default_value}): ").strip()
        if not user_input:
            return default_value
        try:
            value = int(user_input)
            if value > 0: # Basic check for positive integers where applicable
                 return value
            else:
                 print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

# --- Main Execution Block (Command Line Interface) ---
if __name__ == "__main__":

    print("--- DSDeepResearch_Lite CLI ---")
    print("Ensure OPENROUTER_API_KEY and SERPAPI_API_KEY are set as environment variables.")
    if not SERPAPI_API_KEY:
        print("WARNING: SERPAPI_API_KEY not set. Web search will be skipped.")

    # --- Get User Inputs ---
    while True:
        lang_choice = input("Select output language ('en' for English, 'zh' for Chinese) [en]: ").lower().strip()
        if not lang_choice: lang_choice = 'en' # Default to English
        if lang_choice in ['en', 'zh']: break
        else: print("Invalid choice.")

    research_topic = ""
    while not research_topic:
        research_topic = input(f"Enter the research topic or study direction (in {lang_choice}): ").strip()
        if not research_topic: print("Topic cannot be empty.")

    while True:
        study_mode_choice = input("Select study mode ('testing' or 'full') [testing]: ").lower().strip()
        if not study_mode_choice: study_mode_choice = 'testing' # Default
        if study_mode_choice in ['testing', 'full']: break
        else: print("Invalid choice.")

    output_path = input(f"Enter output directory path (Default: '{DEFAULT_OUTPUT_PATH}'): ").strip()
    if not output_path: output_path = DEFAULT_OUTPUT_PATH

    print("\n--- Advanced Parameters (Press Enter to use defaults) ---")
    target_sources = get_int_input("Target Sources", DEFAULT_TARGET_SOURCE_COUNT)
    max_searches = get_int_input("Max Search Attempts", DEFAULT_MAX_SEARCH_ATTEMPTS)
    results_per_q = get_int_input("Results per Query", DEFAULT_SEARCH_RESULTS_PER_QUERY)
    max_refinements = get_int_input("Max Query Refinements", DEFAULT_MAX_QUERY_REFINEMENT_ATTEMPTS)
    max_tk_outline = get_int_input("Max Tokens (Outline)", DEFAULT_MAX_TOKENS_OUTLINE)
    max_tk_synthesis = get_int_input("Max Tokens (Section)", DEFAULT_MAX_TOKENS_SYNTHESIS_PER_SECTION)
    max_tk_query_gen = get_int_input("Max Tokens (Query Gen)", DEFAULT_MAX_TOKENS_QUERY_GENERATION)
    max_tk_query_refine = get_int_input("Max Tokens (Query Refine)", DEFAULT_MAX_TOKENS_QUERY_REFINEMENT)

    print("\n--- Starting Research ---")

    # Call the main research function
    final_data = run_research_process(
        topic=research_topic,
        lang_code=lang_choice,
        output_queue=None, # No queue for CLI version
        target_model_id=DEEPSEEK_MODEL,
        target_sources=target_sources,
        max_searches=max_searches,
        study_mode=study_mode_choice,
        max_refinements=max_refinements,
        max_tokens_outline=max_tk_outline,
        max_tokens_synthesis=max_tk_synthesis,
        search_results_per_q=results_per_q,
        max_tokens_query_gen=max_tk_query_gen,
        max_tokens_query_refine=max_tk_query_refine
    )

    # --- Process Final Output ---
    if final_data:
        lang_strings = LANG_STRINGS[lang_choice]
        print(f"\n\n=== Assembled Document (Draft - {lang_choice.upper()} - {study_mode_choice} mode) ===")

        # Assemble document (copied from worker function for CLI output)
        final_content_warning = "(Content based on gathered sources via SerpApi - General Search)"
        if not SERPAPI_API_KEY or not final_data.get("sources"): final_content_warning = "(Web search skipped or failed; content may be limited)"
        full_text = []
        full_text.append(f"# Research Document: {final_data['topic']}\n")
        full_text.append(f"\n## Outline (Generated from Sources) ({lang_choice.upper()})")
        full_text.append(final_data.get('outline', lang_strings['not_generated']))
        full_text.append(f"\n## Synthesized Content {final_content_warning}")
        processed_sections = list(final_data['synthesis'].keys())
        # Need to re-parse outline here if needed for ordering
        final_outline_sections = parse_outline_sections(final_data['outline'])
        ordered_sections_to_print = [s for s in final_outline_sections if s in processed_sections]
        for section_key in processed_sections:
             if section_key not in ordered_sections_to_print: ordered_sections_to_print.append(section_key)
        if ordered_sections_to_print:
             for section in ordered_sections_to_print:
                  content = final_data['synthesis'].get(section, lang_strings['section_not_generated'])
                  full_text.append(f"\n### {section.strip()}")
                  full_text.append(content)
             sections_processed_count = len(ordered_sections_to_print)
             if study_mode_choice == 'testing' and sections_processed_count < len(final_outline_sections):
                  full_text.append(f"\n\n[Note: Only the first {sections_processed_count} sections were processed in Testing Mode.]")
             elif study_mode_choice == 'full' and sections_processed_count < len(final_outline_sections):
                  full_text.append(f"\n\n[Note: Processed {sections_processed_count}/{len(final_outline_sections)} sections in Full Study Mode.]")
        else: full_text.append("\n[No sections were synthesized in this run.]")
        full_text.append(f"\n\n## {lang_strings['references_heading']}\n")
        if final_data.get("sources"):
            sorted_sources = sorted(final_data["sources"], key=lambda x: x.get('source_url', ''))
            for i, src in enumerate(sorted_sources):
                url = src.get('source_url', 'N/A'); text_snippet = src.get('text', '')[:120]
                full_text.append(f"{i+1}. URL: {url}\n   Info: {text_snippet}...")
        else: full_text.append(lang_strings['no_references'])
        assembled_document = "\n".join(full_text)

        # Print final document to console
        try:
             print(assembled_document)
        except UnicodeEncodeError:
             print(f"\nWarning: Terminal might not fully support UTF-8 for displaying {lang_choice} characters directly.")
             print("Document assembly completed. Check saved files.")

        # Calculate final character count
        final_char_count = len(assembled_document)
        print(f"\n\n=== {lang_strings['final_char_count_label']} (Assembled Document): {final_char_count} ===")

        # --- Save Files ---
        try:
            target_directory = output_path # Use path from input
            os.makedirs(target_directory, exist_ok=True)
            print(f"\n--- Attempting to save results to: {target_directory} ---")

            safe_topic = re.sub(r'[\\/*?:"<>|]', "", research_topic)
            safe_topic = safe_topic.replace(' ', '_').lower()[:50]
            # Add study mode to filename for clarity
            file_suffix = f"{lang_choice}_source_first_{study_mode_choice}"
            base_data_filename = f"research_data_{safe_topic}_{file_suffix}.json"
            base_text_filename = f"research_report_{safe_topic}_{file_suffix}.txt"

            full_data_path = os.path.join(target_directory, base_data_filename)
            full_text_path = os.path.join(target_directory, base_text_filename)

            with open(full_data_path, "w", encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
            print(f"Full research data saved to: {full_data_path}")

            with open(full_text_path, "w", encoding='utf-8') as f:
                f.write(assembled_document)
            print(f"Assembled document saved to: {full_text_path}")

        except Exception as e:
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Error saving files to target directory: {e}")
            print(f"Please check permissions and path: {target_directory}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    else:
        print("\n--- Research process failed to return data. ---")

    print("\n--- Script Finished ---")
