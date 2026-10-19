# Two-Stage RAG Pipeline: Baseline vs. Re-Ranking

This repository contains the code and evaluation report for implementing and comparing two Retrieval-Augmented Generation (RAG) pipelines over a complex document corpus (FIDE Laws of Chess).

## Overview
* **Pipeline 1 (Baseline Retrieval):** Uses a Bi-Encoder (`all-MiniLM-L6-v2`) and a FAISS index for fast, independent vector retrieval based on cosine similarity.
* **Pipeline 2 (Advanced Re-Ranking):** Passes the top 10 baseline candidates through a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to jointly score and re-rank the chunks before LLM generation.

## Repository Contents
* `RAG-assign.ipynb`: The main Jupyter Notebook containing the chunking, embedding, retrieval, cross-encoder re-ranking, and latency profiling code.
* `rag_assignment_report.pdf`: The final report detailing architecture comparisons, specific rank shifts in failure modes, and latency trade-offs.
* `LawsOfChess.pdf`: The reference dataset.

## Key Findings
The Cross-Encoder successfully resolved complex query failure modes (such as hidden rule exceptions and unsupervised venue protocols) by promoting critical context chunks from the bottom of the baseline results (e.g., Rank 5 or 6) directly to Rank 1. This prevented LLM hallucinations and improved factual accuracy, albeit with a slight increase in computational latency.

## Setup
To run the notebook locally:
1. Install the required libraries: `pip install faiss-cpu pypdf sentence-transformers google-genai numpy`
2. Set your Gemini API key as an environment variable: `export GEMINI_API_KEY2="your_key_here"`
3. Execute the cells in `RAG-assign.ipynb`.
