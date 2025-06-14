# Resume Screening Assistance

## Overview

This project provides an AI-powered Resume Screening Assistant built with Streamlit. It enables HR professionals and recruiters to efficiently screen and analyze resumes against a provided job description. The system uses vector embeddings, Pinecone for semantic search, and the LLaMA-3 language model (via Groq) to extract, summarize, and retrieve relevant candidate resumes from uploaded PDF files.

## Features

* Upload and process multiple PDF resumes.
* Enter a job description and retrieve top-K relevant resumes.
* Uses HuggingFace MiniLM model to generate document embeddings.
* Push and retrieve documents to/from Pinecone vector database.
* Dynamically creates and filters results using a unique session ID.
* Summarizes resumes using LLaMA-3 (Groq) for better interpretation.

## Installation

### Clone the repository

```bash
git clone https://github.com/thelakshyadubey/Resume_Screening_Assistance.git
cd Resume_Screening_Assistance
```

### Set up a virtual environment (optional but recommended)

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file with the following keys:

```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
```

## Usage

```bash
streamlit run app.py
```

1. Paste a job description.
2. Specify the number of resumes to return.
3. Upload one or more PDF resumes.
4. Click on **Help me with the analysis** to retrieve relevant matches.
5. View file name, match score, and summarized content of each relevant resume.

## File Structure

```
resume-screening-assistant/
├── app.py               # Main Streamlit interface
├── utils.py             # Helper functions for document parsing, Pinecone operations
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not included in repo)
```

## Author
Lakshya Dubey

## Preview
![image](https://github.com/user-attachments/assets/aea8f656-ef6f-47ed-9eeb-25da739ca518)
![image](https://github.com/user-attachments/assets/9c90619b-68bb-4ea8-a417-e8feb338bdf4)
