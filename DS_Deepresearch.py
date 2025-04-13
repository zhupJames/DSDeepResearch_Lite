
import json
import os
import sys
import time
import re
import requests.exceptions
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import threading
import queue # For thread-safe GUI updates

# --- Configuration & Global Setup ---

# Attempt to load API keys from environment variables
# These MUST be set before running the GUI script.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# --- Constants ---
# LLM Model Configuration (Ensure this matches OpenRouter ID)
DEEPSEEK_MODEL = "deepseek/deepseek-chat"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# SerpApi Configuration
SERPAPI_ENDPOINT = "https://serpapi.com/search"

# LLM API Call Headers
LLM_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}", # Key loaded at start
    "Content-Type": "application/json",
    "HTTP-Referer": "YOUR_APP_URL_OR_NAME", # Optional: Replace with your app's URL or name
    "X-Title": "Deep Research GUI" # Optional: Replace with your app's name
}

# Token Limits (Adjust as needed, lower values for faster testing)
MAX_TOKENS_SYNTHESIS_PER_SECTION = 500
MAX_TOKENS_OUTLINE = 1500
MAX_TOKENS_PLAN = 500
MAX_TOKENS_QUERY_GENERATION = 300
MAX_TOKENS_QUERY_REFINEMENT = 100

# Source Gathering Parameters
TARGET_SOURCE_COUNT = 40 # Target number of sources (may not be reached)
SEARCH_RESULTS_PER_QUERY = 5 # How many results to request from SerpApi per query
MAX_SEARCH_ATTEMPTS = 10 # Limit SerpApi calls
MAX_QUERY_REFINEMENT_ATTEMPTS = 3 # Limit LLM refinement calls within search loop

# --- Language Strings ---
# Dictionary holding UI text and prompt instructions for different languages
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
        'gui_topic_label': "Research Topic:",
        'gui_language_label': "Output Language:",
        'gui_start_button': "Start Research",
        'gui_status_ready': "Ready. Enter topic and language.",
        'gui_status_running': "Research in progress...",
        'gui_status_done': "Research process finished.",
        'gui_status_error': "Error occurred.",
        'gui_keys_missing_title': "API Keys Missing",
        'gui_keys_missing_msg': "Error: OPENROUTER_API_KEY or SERPAPI_API_KEY environment variable not set.\nPlease set them before running.\nWeb search requires SerpApi key.",
        'gui_serpapi_missing_warning': "WARNING: SerpApi key not provided. Web search step will be skipped.",
    },
    'zh': {
        'system_role': "You are an advanced research assistant responding in Chinese (简体中文).",
        'plan_instruction': "Please respond concisely in English.", # Plan is always English
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
        'gui_topic_label': "研究主题:",
        'gui_language_label': "输出语言:",
        'gui_start_button': "开始研究",
        'gui_status_ready': "就绪。请输入主题和语言。",
        'gui_status_running': "研究进行中...",
        'gui_status_done': "研究过程结束。",
        'gui_status_error': "发生错误。",
        'gui_keys_missing_title': "API密钥缺失",
        'gui_keys_missing_msg': "错误：未设置 OPENROUTER_API_KEY 或 SERPAPI_API_KEY 环境变量。\n请在运行前设置它们。\n网络搜索需要 SerpApi 密钥。",
        'gui_serpapi_missing_warning': "警告：未提供 SerpApi 密钥。将跳过网络搜索步骤。",
    }
}

# --- Core API Interaction Function (for LLM) ---
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
    # Check if OpenRouter key is available (essential for LLM calls)
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured."

    # Select system message based on language
    system_message_base = LANG_STRINGS[lang_code]['system_role']
    system_message = f"""
    {system_message_base} Your primary goals are:
    1. Accuracy & Detail
    2. Instruction Adherence (including language: {lang_code})
    3. Consistency
    4. Clarity
    5. Conditional Citation (follow instructions in user prompt)
    """

    # Prepare payload for the API request
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    # Log the attempt
    print(f"\n--- Sending Prompt to LLM {model} via OpenRouter (Lang: {lang_code}) ---")
    print(f"Prompt (first 300 chars): {prompt[:300]}...")
    print(f"(Requesting max_tokens: {max_tokens})")

    response = None # Initialize response variable
    try:
        # Make the API call using requests library
        print(f"--- Making requests.post call to {OPENROUTER_API_URL} (Timeout={180}s) ---")
        response = requests.post(
            OPENROUTER_API_URL, headers=LLM_HEADERS, json=payload, timeout=180
        )
        print(f"--- requests.post call returned (Status: {response.status_code}) ---")

        # Check for HTTP errors (e.g., 4xx, 5xx)
        if response.status_code != 200:
            print(f"Error Response Text: {response.text}") # Log raw error
            response.raise_for_status() # Raise an exception for bad status codes

        # Parse the JSON response if status code is 200
        result = response.json()

        # Extract the generated content, checking the expected structure
        if "choices" in result and result["choices"] and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
            content = result["choices"][0]["message"]["content"]
            if "usage" in result: print(f"--- API Usage: {result['usage']} ---") # Log token usage if available
            return content.strip() # Return the useful content
        else:
            # Handle cases where the response format is unexpected
            print("Warning: API response status 200 but did not contain expected content path.")
            print("Full response:", json.dumps(result, indent=2))
            if "error" in result:
                 error_message = result['error'].get('message', 'No message provided')
                 try: # Attempt to decode potential unicode escapes in error messages
                     if '\\u' in error_message: error_message = error_message.encode('utf-8').decode('unicode_escape')
                 except Exception: pass
                 print(f"API Error Message: {error_message}")
                 return f"Error: API returned an error - {result['error'].get('code', 'Unknown code')}"
            return "Error: Unexpected API response format"

    # --- Specific Error Handling ---
    except requests.exceptions.Timeout:
        # Handle request timeout
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error: LLM API request timed out after 180 seconds.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        # Handle other request errors (connection, HTTP errors raised by raise_for_status)
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error: LLM API request failed: {e}")
        error_details = ""
        error_text = ""
        # Try to get more details from the response object if it exists
        if response is not None:
             try:
                error_details = response.json() # Try parsing JSON error first
                if "error" in error_details:
                     err_data = error_details["error"]
                     print(f"API Error Code: {err_data.get('code', 'N/A')}")
                     print(f"API Error Message: {err_data.get('message', 'N/A')}")
                     # Check for common, informative error codes
                     if err_data.get('code') == "invalid_api_key": print("Error Detail: Invalid API Key.")
                     elif "context_length_exceeded" in err_data.get("code", ""): print("Error Detail: Context length exceeded.")
                elif "rate limit" in str(error_details).lower(): print("Error Detail: Rate limit likely exceeded.")
             except json.JSONDecodeError: # If JSON fails, use raw text
                error_text = response.text
                error_details = error_text
                if "rate limit" in error_text.lower(): print("Error Detail: Rate limit likely exceeded.")
        print(f"Full Error details (if available): {error_details}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return f"Error: API request failed ({e})"
    except json.JSONDecodeError:
        # Handle error if JSON parsing fails even on a 200 OK response
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Error: Could not decode JSON response from LLM API, even though status was OK.")
        print("Response text:", response.text if response is not None else "N/A")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return "Error: Invalid JSON response"
    except Exception as e:
        # Catch any other unexpected exceptions during the API call
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"An unexpected error occurred during LLM API call: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return f"Error: An unexpected error occurred ({e})."


# --- Helper Function (Implemented External Search using SerpApi) ---
def perform_external_search(query: str):
    """
    Performs a general web search using the SerpApi Google Search API.
    Uses hl=en for searching to maximize results from English-centric sources like arXiv.
    Returns a dictionary with 'text' (snippet+title) and 'source_url', or None if failed.

    Args:
        query (str): The search query string (expected to be in English).

    Returns:
        dict or None: Dictionary containing 'text' and 'source_url' on success, None on failure.
    """
    # Check if SerpApi key is available
    if not SERPAPI_API_KEY:
        print("--- SerpApi API Key not provided. Skipping web search. ---")
        return None # Cannot perform search without the key

    # Always use English interface for potentially better technical results
    hl_param = 'en'

    # Parameters for the SerpApi request
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": SEARCH_RESULTS_PER_QUERY, # Get a few results per query
        "hl": hl_param,                  # Set language interface
        "engine": "google"               # Specify the search engine
    }

    # Log the search attempt
    print(f"--- Calling SerpApi Search ---")
    print(f"--- Query: {query} ---")
    print(f"--- Lang (hl): {hl_param} ---")

    response = None # Initialize response
    try:
        # Make the GET request to SerpApi
        response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=60) # 60s timeout
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        # Parse the JSON response
        results = response.json()

        # Check if 'organic_results' exist and are not empty
        if "organic_results" in results and results["organic_results"]:
            first_result = results["organic_results"][0] # Get the top organic result
            snippet = first_result.get('snippet', 'No snippet available.')
            url = first_result.get('link')
            title = first_result.get('title', 'Unknown Source')

            if url: # Only consider the result valid if it has a URL
                 print(f"--- SerpApi Search Success. Top result: {url} ---")
                 # Combine title and snippet for better context downstream
                 return {'text': f"Title: {title}\nSnippet: {snippet}", 'source_url': url}
            else:
                 # If URL is missing, treat as no result found
                 print("--- SerpApi Search Warning: Top result missing URL. ---")
                 return {'text': f"Title: {title}\nSnippet: {snippet}", 'source_url': None} # Return text but no URL
        else:
            # Handle cases where no organic results are returned
            print("--- SerpApi Search returned no 'organic_results'. ---")
            if "error" in results: # Check if SerpApi itself reported an error (e.g., "Google hasn't returned any results...")
                 print(f"--- SerpApi returned an error message: {results['error']} ---")
            return None # Indicate no usable results found

    # --- Specific Error Handling for Search ---
    except requests.exceptions.Timeout:
        print("--- Error: SerpApi request timed out. ---")
        return None
    except requests.exceptions.RequestException as e:
        print(f"--- Error: SerpApi request failed: {e} ---")
        # Log response body if available, helps debug API key or parameter issues
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
def extract_text_from_pdf(pdf_url: str):
    """
    *** USER IMPLEMENTATION REQUIRED ***
    This function should download the PDF content from the given URL
    and extract text using a library like PyPDF2 or pdfminer.six.

    Args:
        pdf_url (str): The URL of the PDF file.

    Returns:
        str or None: The extracted text (potentially truncated) or None if extraction fails.
    """
    print(f"--- [Placeholder] Attempting to download and extract text from PDF: {pdf_url} ---")
    # Example Implementation (Requires: pip install requests PyPDF2)
    # try:
    #     import requests
    #     import PyPDF2
    #     from io import BytesIO
    #
    #     headers = {'User-Agent': 'Mozilla/5.0'} # Add a user agent
    #     response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
    #     response.raise_for_status()
    #
    #     content_type = response.headers.get('content-type', '').lower()
    #     if 'application/pdf' not in content_type:
    #         print(f"Warning: URL content type is not PDF ({content_type}). Skipping extraction.")
    #         return None
    #
    #     pdf_file = BytesIO()
    #     # Read content carefully, handle potential large files
    #     content_length = 0
    #     max_pdf_size = 20 * 1024 * 1024 # Limit PDF size to 20MB
    #     for chunk in response.iter_content(chunk_size=8192):
    #         content_length += len(chunk)
    #         if content_length > max_pdf_size:
    #             print(f"Error: PDF file exceeds size limit ({max_pdf_size} bytes). Skipping.")
    #             return None
    #         pdf_file.write(chunk)
    #     pdf_file.seek(0)
    #
    #     try:
    #         reader = PyPDF2.PdfReader(pdf_file)
    #         text = ""
    #         num_pages_to_extract = 10 # Limit pages to prevent excessive processing
    #         for i, page in enumerate(reader.pages):
    #             if i >= num_pages_to_extract:
    #                 print(f"--- Limiting PDF extraction to first {num_pages_to_extract} pages. ---")
    #                 break
    #             page_text = page.extract_text()
    #             if page_text:
    #                  text += page_text + "\n"
    #
    #         if text:
    #              print(f"--- Successfully extracted text from PDF (length: {len(text)}) ---")
    #              return text[:10000] # Return first 10k chars
    #         else:
    #              print(f"--- Warning: PyPDF2 extracted no text from {pdf_url} ---")
    #              return None
    #     except Exception as pdf_err: # Catch PyPDF2 specific errors
    #         print(f"--- Error parsing PDF content from {pdf_url} with PyPDF2: {pdf_err} ---")
    #         return None
    #
    # except ImportError:
    #      print("--- PDF Extraction Skipped: `requests` or `PyPDF2` library not installed. ---")
    #      return None
    # except requests.exceptions.RequestException as req_e:
    #      print(f"--- Error downloading PDF {pdf_url}: {req_e} ---")
    #      return None
    # except Exception as e:
    #     print(f"--- Error extracting text from PDF {pdf_url}: {e} ---")
    #     return None
    print("--- PDF Extraction not implemented. Returning None. ---")
    return None # Placeholder returns None


# --- Helper Function (Outline Parsing - Needs Improvement) ---
def parse_outline_sections(outline_text):
    """
    Parses outline text to extract section identifiers/titles.
    Uses regex for common English and Chinese formats. Needs improvement for robustness.

    Args:
        outline_text (str): The string containing the generated outline.

    Returns:
        list: A list of strings, each representing a potential section header.
    """
    # Regex attempts to match various common outline formats
    pattern = r"^\s*(?:[IVXLCDM]+\.|[A-Z]\.|[0-9]+\.|[a-z]\.|[一二三四五六七八九十]+[、．\.])\s+.*"
    sections = []
    lines = outline_text.splitlines()
    for line in lines:
        trimmed_line = line.strip()
        # Check if the line matches the pattern
        if re.match(pattern, trimmed_line):
             # Further check: ensure it's not just an identifier by splitting once
             parts = trimmed_line.split(None, 1)
             # Keep if there's text after identifier OR if the identifier line itself is reasonably long
             if (len(parts) > 1 and len(parts[1].strip()) > 0) or \
                (len(parts) == 1 and len(parts[0]) > 5):
                  sections.append(trimmed_line)
    # Fallback if no standard patterns are found
    if not sections:
         sections = [line.strip() for line in lines if line.strip() and len(line.strip()) > 1] # Use any non-short, non-empty line
         print("Warning: Basic outline parsing failed to find standard patterns. Using non-empty lines as sections.")
    print(f"Parsed {len(sections)} potential sections from outline using basic regex.")
    return sections


# --- Research Workflow Simulation (Source-First with SerpApi, General Search) ---
def run_research_process(topic, lang_code='en', output_queue=None):
    """
    Main research process function, adapted for GUI.

    Args:
        topic (str): The research topic entered by the user.
        lang_code (str): The selected language code ('en' or 'zh').
        output_queue (queue.Queue, optional): Queue to send status/output messages to the GUI. Defaults to None.
    """

    # Helper to send messages to GUI queue or print to console
    def log_message(message):
        if output_queue:
            output_queue.put(message)
        else:
            print(message) # Fallback to console if no queue

    log_message(f"\n=== Starting Research Process for: {topic} (Language: {lang_code}, Workflow: Source-First/SerpApi General Search) ===")
    lang_strings = LANG_STRINGS[lang_code]

    research_data = {
        "topic": topic,
        "sources": [],
        "outline": "",
        "synthesis": {},
        "estimated_char_count": 0
    }

    target_model_id = DEEPSEEK_MODEL
    log_message(f"--- Using LLM Model ID: {target_model_id} ---")

    # --- Step 1: Source Gathering ---
    log_message(f"\n[Step 1: Gathering Sources via SerpApi (Max Attempts: {MAX_SEARCH_ATTEMPTS}, Search lang: en)]")

    # 1a. Generate OPTIMIZED Search Queries using LLM
    log_message("--- Generating OPTIMIZED search queries using LLM ---")
    query_gen_prompt = f"""
    Based on the research topic "{topic}", generate a list of 5 diverse and optimized English Google search queries designed to find the most relevant and high-quality sources (like academic papers, reputable web resources).
    Focus on precision, key technical terms, synonyms, and potential variations that capture the core concepts effectively for maximizing relevant search results.
    Output ONLY the list of queries, one query per line. Do not add numbering or introductory text.
    """
    # Always generate English queries for search
    llm_generated_queries_str = call_llm_api(query_gen_prompt, lang_code='en', model=target_model_id, max_tokens=MAX_TOKENS_QUERY_GENERATION)

    search_queries = []
    if not llm_generated_queries_str.startswith("Error:") and llm_generated_queries_str:
        search_queries = [q.strip() for q in llm_generated_queries_str.splitlines() if q.strip()]
        search_queries = [q for q in search_queries if len(q) > 3 and not q.startswith("Here") and not q.startswith("1.")] # Basic filter
        log_message(f"--- LLM generated {len(search_queries)} initial search queries: ---")
        for q in search_queries: log_message(f"  - {q}")
    else:
        log_message("--- Warning: Failed to generate search queries via LLM. Using basic topic query. ---")
        search_queries = []

    if not search_queries: # Fallback if LLM fails or returns empty/bad list
         log_message("--- Using fallback basic queries ---")
         search_queries = [f'"{topic}"', f'"{topic}" review', f'"{topic}" survey']


    # 1b. Perform Searches using Generated Queries with Refinement on Failure
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

        search_result = perform_external_search(current_query) # Calls SerpApi

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
                    search_succeeded = True # Treat duplicate as non-failure for cycling
            else:
                 log_message(f"--- Skipping search engine result page: {source_url} ---")
                 search_succeeded = False
        else:
            log_message(f"--- Search attempt yielded no usable result or URL. ---")
            if search_result is None and SERPAPI_API_KEY:
                 log_message("--- Search failed. Check SerpApi key and service status. ---")
            search_succeeded = False

        # LLM Query Refinement on Failure
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
                 # Replace the current query in the list to try it next
                 search_queries[query_index % len(search_queries)] = new_query_str
                 # Don't advance query_index, so we try the refined query immediately
                 # query_index += 0 # No change needed here, will retry same effective index
            else:
                 log_message("--- LLM failed to generate refinement query. Continuing with next original query. ---")
                 query_index += 1 # Move to next original query index
        else:
             # If search succeeded OR refinement limit reached OR max attempts reached, move to next query index
             query_index += 1

        if len(gathered_sources_dict) >= TARGET_SOURCE_COUNT:
            log_message(f"--- Reached target source count ({TARGET_SOURCE_COUNT}). Stopping search. ---")
            break

        time.sleep(1.5) # API delay

    research_data["sources"] = list(gathered_sources_dict.values())
    log_message(f"\n--- Finished Source Gathering after {query_attempts} attempts: Collected {len(research_data['sources'])} unique sources. ---")

    if not research_data["sources"]:
        log_message("Could not gather any sources. Unable to proceed with source-based outline/synthesis.")
        return None # Indicate failure

    # Prepare source material summary for outline generation
    source_summary_for_outline = ""
    for i, src in enumerate(research_data["sources"]):
        source_summary_for_outline += f"Source {i+1} ({src.get('source_url', 'N/A')}):\n{src.get('text', 'N/A')[:500]}...\n\n"


    # --- Step 2: Outline Generation ---
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
        return None # Indicate failure
    log_message(f"\nGenerated Outline (Based on Sources, {lang_code}):")
    log_message(research_data["outline"])
    time.sleep(1)


    # --- Step 3: Section-by-Section Synthesis ---
    log_message(f"\n[Step 3: Synthesizing Content Section by Section ({lang_code})]")
    current_outline_sections = parse_outline_sections(research_data["outline"])

    # Limit sections processed for testing (can be removed for full run)
    SECTIONS_TO_PROCESS_LIMIT = 3
    sections_to_synthesize = current_outline_sections[:SECTIONS_TO_PROCESS_LIMIT]
    log_message(f"--- Processing only the first {len(sections_to_synthesize)} outline points for synthesis (Testing Mode) ---")

    if not sections_to_synthesize:
        log_message("Could not parse sections from the outline or limit resulted in zero sections. Cannot synthesize content.")
    else:
        total_synthesized_chars = 0
        for i, section_id in enumerate(sections_to_synthesize):
            log_message(f"\n--- Synthesizing section {i+1}/{len(sections_to_synthesize)}: '{section_id}' ---")

            # Prepare source info relevant to this section (provide all sources for now)
            relevant_info_parts = []
            source_mapping = {}
            for idx, src in enumerate(research_data["sources"]):
                 source_num = idx + 1
                 url = src.get('source_url', 'N/A')
                 text_snippet = src.get('text', 'N/A')[:1500]
                 relevant_info_parts.append(f"Source {source_num}:\n{text_snippet}\n(URL: {url})")
                 source_mapping[source_num] = url
            relevant_info_prompt_block = "\n\n".join(relevant_info_parts)

            citation_instruction = lang_strings[f'citation_instruction_{lang_code}']

            synthesis_prompt = f"""
            Objective: Write a **concise but informative** section in {lang_code} for the research document on '{research_data['topic']}', exclusively covering the content for the outline point: "{section_id}".
            Base your writing *primarily* on the provided source materials gathered from web search.

            **Full Current Outline (for context ONLY):**
            {research_data['outline']}
            ---

            **Section to Write:** {section_id}

            **Provided Source Materials (Use these to write the section):**
            ---
            {relevant_info_prompt_block[:10000]}
            ---

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
                log_message(synthesized_content[:300] + "...") # Log snippet
            else:
                log_message(f"Failed to synthesize content for section '{section_id}'.")
                research_data["synthesis"][section_id] = f"Error: Failed to generate content for {section_id}."

            research_data["estimated_char_count"] = total_synthesized_chars
            log_message(f"{lang_strings['char_count_total_label']}: {total_synthesized_chars}")
            time.sleep(2)

    log_message(f"\n=== Research Process for: {topic} Completed (Source-First Mode) ===")
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
         if len(ordered_sections_to_print) < len(final_outline_sections):
              full_text.append(f"\n\n[Note: Only the first {len(ordered_sections_to_print)} sections were processed in this short run.]")
    else:
         full_text.append("\n[No sections were synthesized in this run.]")

    # Add Collected Sources List
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

    # Put final result in queue for GUI
    if output_queue:
        output_queue.put("\n\n=== Final Assembled Document ===\n")
        output_queue.put(assembled_document)
        # Optionally save files here as well, or let GUI handle saving
        # Consider saving logic might block GUI thread if run directly here
        # save_files(research_data, assembled_document, lang_code) # Example call
    else:
        # Fallback if not run from GUI
        print("\n\n=== Final Assembled Document ===\n")
        print(assembled_document)
        # save_files(research_data, assembled_document, lang_code) # Example call

    return research_data # Return data for potential further use


# --- GUI Application Class ---
class ResearchApp:
    def __init__(self, root):
        """Initializes the Tkinter GUI application."""
        self.root = root
        self.root.title("Deep Research Assistant GUI")
        # Set a minimum size
        self.root.minsize(600, 500)

        # Configure styles
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam') # Use a modern theme if available

        # Set default font
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=10)
        text_font = font.nametofont("TkTextFont")
        text_font.configure(size=10)
        self.root.option_add("*Font", default_font)

        # --- Variables ---
        self.topic_var = tk.StringVar()
        self.lang_var = tk.StringVar(value='en') # Default to English
        self.status_var = tk.StringVar(value="Initializing...")

        # Queue for communication between research thread and GUI
        self.output_queue = queue.Queue()

        # --- Create Widgets ---
        self.create_widgets()

        # --- Check API Keys on Startup ---
        self.check_api_keys()

    def create_widgets(self):
        """Creates and lays out the GUI widgets."""

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        # Topic Label and Entry
        topic_label = ttk.Label(input_frame, text="Research Topic:")
        topic_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.topic_entry = ttk.Entry(input_frame, textvariable=self.topic_var, width=60)
        self.topic_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Language Label and Radio Buttons
        lang_label = ttk.Label(input_frame, text="Output Language:")
        lang_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        lang_frame = ttk.Frame(input_frame)
        lang_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        en_radio = ttk.Radiobutton(lang_frame, text="English", variable=self.lang_var, value='en')
        en_radio.pack(side=tk.LEFT, padx=5)
        zh_radio = ttk.Radiobutton(lang_frame, text="Chinese (简体)", variable=self.lang_var, value='zh')
        zh_radio.pack(side=tk.LEFT, padx=5)

        # Start Button
        self.start_button = ttk.Button(input_frame, text="Start Research", command=self.start_research_thread)
        self.start_button.grid(row=2, column=0, columnspan=2, pady=10)

        # --- Output Frame ---
        output_frame = ttk.LabelFrame(main_frame, text="Output Log & Results", padding="10")
        output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.rowconfigure(1, weight=1) # Allow output frame to expand

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=20, width=80, state='disabled')
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        # --- Status Bar ---
        status_bar = ttk.Frame(main_frame, relief=tk.SUNKEN, padding="2 5 2 5")
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=(5,0))
        self.status_label = ttk.Label(status_bar, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(fill=tk.X)

    def log_to_gui(self, message):
        """Appends a message to the output text area in a thread-safe way."""
        self.output_text.configure(state='normal') # Enable writing
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.configure(state='disabled') # Disable writing
        self.output_text.see(tk.END) # Scroll to the end
        self.root.update_idletasks() # Process GUI events

    def check_api_keys(self):
        """Checks for API keys on startup and updates status/button."""
        missing_keys = []
        if not OPENROUTER_API_KEY:
            missing_keys.append("OPENROUTER_API_KEY")
        if not SERPAPI_API_KEY:
            missing_keys.append("SERPAPI_API_KEY (needed for web search)")

        if missing_keys:
            error_msg = f"Error: Required environment variable(s) not set: {', '.join(missing_keys)}.\nPlease set them and restart."
            self.status_var.set(error_msg)
            self.log_to_gui(error_msg)
            self.start_button.configure(state='disabled') # Disable start button
            messagebox.showerror("API Key Error", error_msg)
            return False
        else:
            self.status_var.set(LANG_STRINGS[self.lang_var.get()]['gui_status_ready'])
            self.start_button.configure(state='normal')
            return True

    def start_research_thread(self):
        """Starts the research process in a separate thread."""
        if not self.check_api_keys(): # Re-check keys before starting
             return

        topic = self.topic_var.get().strip()
        lang = self.lang_var.get()

        if not topic:
            messagebox.showwarning("Input Missing", "Please enter a research topic.")
            return

        # Disable button, clear output, update status
        self.start_button.configure(state='disabled')
        self.output_text.configure(state='normal')
        self.output_text.delete('1.0', tk.END)
        self.output_text.configure(state='disabled')
        self.status_var.set(LANG_STRINGS[lang]['gui_status_running'])
        self.log_to_gui(f"Starting research for topic: '{topic}' in language: {lang}")
        if not SERPAPI_API_KEY:
            self.log_to_gui(LANG_STRINGS[lang]['gui_serpapi_missing_warning'])


        # Create and start the background thread
        self.research_thread = threading.Thread(
            target=research_worker, # Function to run in thread
            args=(topic, lang, self.output_queue), # Arguments for the worker
            daemon=True # Allows main program to exit even if thread is running
        )
        self.research_thread.start()

        # Start checking the queue for updates
        self.root.after(100, self.process_queue)

    def process_queue(self):
        """Processes messages from the research thread queue to update the GUI."""
        try:
            # Get all messages currently in the queue (non-blocking)
            while True:
                message = self.output_queue.get_nowait()
                if message == "===RESEARCH_COMPLETE===":
                    self.status_var.set(LANG_STRINGS[self.lang_var.get()]['gui_status_done'])
                    self.start_button.configure(state='normal') # Re-enable button
                    # Optionally save files here from GUI thread
                    # save_files(...)
                elif message == "===RESEARCH_ERROR===":
                     self.status_var.set(LANG_STRINGS[self.lang_var.get()]['gui_status_error'])
                     self.start_button.configure(state='normal') # Re-enable button
                else:
                    self.log_to_gui(str(message)) # Display message in text area
        except queue.Empty:
            pass # No messages currently in queue

        # If the thread is still alive, schedule another check
        if self.research_thread.is_alive():
            self.root.after(100, self.process_queue) # Check again in 100ms
        else:
            # Thread finished, final check just in case
            try:
                 while True: # Process any remaining messages
                      message = self.output_queue.get_nowait()
                      if message == "===RESEARCH_COMPLETE===":
                           self.status_var.set(LANG_STRINGS[self.lang_var.get()]['gui_status_done'])
                      elif message == "===RESEARCH_ERROR===":
                           self.status_var.set(LANG_STRINGS[self.lang_var.get()]['gui_status_error'])
                      else:
                           self.log_to_gui(str(message))
            except queue.Empty:
                 pass
            # Ensure button is re-enabled if thread finished but completion message wasn't last
            if self.start_button['state'] == 'disabled':
                 self.start_button.configure(state='normal')
                 if not self.status_var.get().endswith("finished.") and not self.status_var.get().endswith("Error occurred."):
                      self.status_var.set(LANG_STRINGS[self.lang_var.get()]['gui_status_done'] + " (Thread ended)")



# --- Worker Function (Runs in Background Thread) ---
def research_worker(topic, lang_code, output_queue):
    """
    Wrapper function to run the research process and put output/status in the queue.
    This function runs in the background thread.
    """
    # Redirect print statements within this thread to the queue
    class QueueLogger:
        def __init__(self, queue):
            self.queue = queue
        def write(self, message):
            self.queue.put(message.rstrip()) # Remove trailing newline if print adds one
        def flush(self):
            pass # Needed for stdout interface

    original_stdout = sys.stdout # Store original stdout
    sys.stdout = QueueLogger(output_queue) # Redirect

    try:
        # Run the main research process
        final_data = run_research_process(topic, lang_code, output_queue) # Pass queue

        if final_data:
            # Indicate completion (GUI will handle final assembly display if needed)
            output_queue.put("===RESEARCH_COMPLETE===")
            # --- File saving logic moved to main thread or needs careful handling ---
            # save_files(final_data, assembled_document, lang_code) # Be careful with GUI blocking
        else:
            output_queue.put("Research process returned no data.")
            output_queue.put("===RESEARCH_ERROR===")

    except Exception as e:
        # Log any unexpected error during the process
        output_queue.put(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        output_queue.put(f"An error occurred in the research worker thread: {e}")
        import traceback
        output_queue.put(traceback.format_exc()) # Put full traceback in queue
        output_queue.put(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        output_queue.put("===RESEARCH_ERROR===")
    finally:
        sys.stdout = original_stdout # Restore original stdout

# --- Main Execution ---
if __name__ == "__main__":
    # This block now only sets up and runs the GUI
    root = tk.Tk()
    app = ResearchApp(root)
    root.mainloop() # Start the Tkinter event loop

