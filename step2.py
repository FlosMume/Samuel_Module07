"""
Step 2: Data Sampling & Preparation (Standalone, No Dependencies)

Purpose:
    - Create a fresh dataset of academic papers from arXiv using the arXiv API.
    - Sample 100 real research papers (metadata only: title, abstract, ID, etc.).
    - Extract and save their abstracts for use in synthetic Q&A generation.
    - Output structured files for downstream steps in fine-tuning.

Why This Step Matters:
    - High-quality, domain-specific data is essential for effective fine-tuning.
    - Using real academic abstracts ensures our synthetic Q&A data reflects actual scholarly language and complexity.
    - This step decouples the project from prior weeks, making it self-contained and reproducible.

Outputs:
    1. data/arxiv_papers.jsonl        -> Full list of fetched papers (one JSON per line)
    2. data/selected_papers.txt       -> List of 100 paper IDs (for traceability)
    3. data/paper_abstracts/*.txt     -> One file per paper containing its abstract

Dependencies:
    - pip install arxiv python-dotenv
    - Internet connection to query arXiv API
"""

# ----------------------------
# Import required libraries
# ----------------------------

import os                  # For interacting with the operating system (e.g., file paths, directories)
import json                # For reading/writing JSON-formatted data
import logging             # For structured logging of progress and errors
from pathlib import Path   # For robust, cross-platform path handling
import arxiv               # Official Python wrapper for the arXiv API (https://github.com/lukasschwab/arxiv.py)
import re                  # For cleaning filenames by removing invalid characters
from datetime import datetime  # Not used here yet, but useful for future timestamping

# ----------------------------
# Configure Logging
# ----------------------------

# Ensure the logs directory exists before initializing the file handler
os.makedirs("logs", exist_ok=True)

# Set up logging to both console and a log file for debugging and tracking
logging.basicConfig(
    level=logging.INFO,  # Log all INFO-level messages and above
    format='%(asctime)s - %(levelname)s - %(message)s',  # Timestamp, level, message
    handlers=[
        logging.FileHandler("logs/sampling_log.txt"),  # Save logs to file
        logging.StreamHandler()                       # Also print logs to terminal
    ]
)
logger = logging.getLogger(__name__)  # Create a logger specific to this script

# ----------------------------
# Configuration Constants
# ----------------------------

# Define main project directory (WSL path)
WORKING_DIR = Path("/home/myunix/llm_projects/week7hw")

# Define data subdirectories and files
DATA_DIR = WORKING_DIR / "data"
PAPER_ABSTRACTS_DIR = DATA_DIR / "paper_abstracts"
ARXIV_DATASET_PATH = DATA_DIR / "arxiv_papers.jsonl"
SELECTED_PAPERS_TXT = DATA_DIR / "selected_papers.txt"

# Number of papers to fetch from arXiv
NUM_PAPERS = 100

# List of relevant arXiv categories (to ensure diversity across AI, ML, NLP, etc.)
CATEGORIES = [
    "cs.AI",     # Artificial Intelligence
    "cs.LG",     # Machine Learning
    "cs.CL",     # Computation and Language
    "stat.ML",   # Statistics: Machine Learning
    "physics.comp-ph",  # Computational Physics
    "math.NA"    # Numerical Analysis
]

# Keywords related to modern LLMs and AI research
KEYWORDS = [
    "language model", "neural network", "fine-tuning", 
    "transformer", "LLM", "machine learning", "attention mechanism"
]

# ----------------------------
# Helper Function: Clean Filenames
# ----------------------------

def clean_filename(filename):
    """
    Remove or replace characters that are invalid in file systems (Windows/Linux).
    This prevents errors when saving abstracts with special characters in titles.
    
    Args:
        filename (str): Original filename or paper ID
    
    Returns:
        str: Safe filename with only valid characters
    """
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename)

# ----------------------------
# Main Function: Fetch Papers from arXiv API
# ----------------------------

def fetch_arxiv_papers(n=100):
    """
    Fetch n recent academic papers from arXiv using diverse search queries.
    
    Strategy:
        - Combine keywords with different categories to maximize diversity.
        - Use arXiv API to get metadata: title, abstract, ID, categories, publication date.
        - Avoid duplicates using a set to track seen paper IDs.
    
    Args:
        n (int): Number of unique papers to fetch
    
    Returns:
        list of dict: Each dict contains paper metadata (id, title, abstract, etc.)
    """
    papers = []  # Store fetched papers
    seen_ids = set()  # Track already-fetched paper IDs to avoid duplicates

    # Generate diverse search queries by combining categories and keywords
    queries = [
        f"cat:({cat}) AND ({kw})"
        for cat in CATEGORIES
        for kw in KEYWORDS
    ]

    logger.info(f"Starting to fetch {n} papers using {len(queries)} diverse queries...")

    # Iterate through queries until we have enough papers
    for query in queries:
        if len(papers) >= n:
            break  # Stop if we've collected enough

        try:
            # Configure search: limit results, sort by recent submission
            search = arxiv.Search(
                query=query,
                max_results=20,  # Fetch up to 20 per query
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            # Execute search and process each result
            for result in search.results():
                paper_id = result.entry_id.split('/')[-1]  # Extract clean ID (e.g., 2401.12345)

                # Skip if already seen or limit reached
                if paper_id in seen_ids or len(papers) >= n:
                    continue

                seen_ids.add(paper_id)

                # Clean and format paper data
                paper_data = {
                    "id": paper_id,
                    "paper_id": paper_id,
                    "title": result.title.replace('\n', ' ').strip(),
                    "abstract": result.summary.replace('\n', ' ').strip(),  # 'summary' = abstract
                    "published": str(result.published),  # ISO format datetime
                    "categories": " ".join(result.categories),
                    "url": result.entry_id  # Full arXiv URL
                }

                papers.append(paper_data)
                logger.debug(f"Fetched: '{paper_data['title']}' [{paper_data['id']}]")

        except Exception as e:
            logger.warning(f"Error during query '{query}': {e}")

    logger.info(f"✅ Successfully fetched {len(papers)} unique papers from arXiv.")
    return papers

# ----------------------------
# Helper Functions: Save Data
# ----------------------------

def save_dataset(papers, output_path):
    """
    Save the full list of papers to a JSONL (JSON Lines) file.
    JSONL is ideal for large datasets: one JSON object per line, easy to stream.

    Args:
        papers (list): List of paper dictionaries
        output_path (Path): File path to save the dataset
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')
    logger.info(f"💾 Dataset saved to: {output_path}")

def save_paper_ids(papers, output_path):
    """
    Save only the paper IDs to a plain text file (one ID per line).
    Useful for tracking which papers were selected.

    Args:
        papers (list): List of paper dictionaries
        output_path (Path): Path to save the list
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(paper["id"] + "\n")
    logger.info(f"📋 Selected paper IDs saved to: {output_path}")

def save_abstracts(papers, output_dir):
    """
    Save each paper's abstract to a separate .txt file named after its ID.

    Args:
        papers (list): List of paper dictionaries
        output_dir (Path): Directory to save abstract files
    """
    for paper in papers:
        safe_id = clean_filename(paper["id"])
        abstract = paper["abstract"]
        output_path = output_dir / f"{safe_id}_abstract.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(abstract)

    logger.info(f"📄 Abstracts saved to directory: {output_dir}")

# ----------------------------
# Main Execution Flow
# ----------------------------

def main():
    """
    Orchestrate the entire Step 2 workflow:
        1. Create necessary directories
        2. Fetch real arXiv papers
        3. Save dataset, ID list, and abstracts
    """
    logger.info("🚀 Starting Step 2: Data Sampling & Preparation (Standalone Mode)")

    # Ensure all required directories exist
    DATA_DIR.mkdir(exist_ok=True)
    PAPER_ABSTRACTS_DIR.mkdir(exist_ok=True)

    # 1. Fetch 100 real arXiv papers using diverse queries
    papers = fetch_arxiv_papers(NUM_PAPERS)

    # Safety check: ensure we got data
    if len(papers) == 0:
        logger.error("❌ No papers were fetched. Check internet connection or arXiv API availability.")
        return

    # 2. Save full dataset in JSONL format
    save_dataset(papers, ARXIV_DATASET_PATH)

    # 3. Save list of selected paper IDs
    save_paper_ids(papers, SELECTED_PAPERS_TXT)

    # 4. Save each abstract as a separate text file
    save_abstracts(papers, PAPER_ABSTRACTS_DIR)

    # Final success message
    logger.info("🎉 Step 2 completed successfully!")
    logger.info(f"📄 Total papers processed: {len(papers)}")
    logger.info(f"📁 Dataset: {ARXIV_DATASET_PATH}")
    logger.info(f"📋 Paper IDs: {SELECTED_PAPERS_TXT}")
    logger.info(f"📂 Abstracts: {PAPER_ABSTRACTS_DIR}/")

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    main()