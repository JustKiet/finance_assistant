# **FINAST: A Finance Assistant**

## ***!VERY IMPORTANT!***

The `OpenAIPDFParser` currently **ONLY WORKS** on a PDF file with tabular data was pre-printed! The parser ***DOES NOT WORK*** on the entire finance report! Refer to `example_input.pdf` for the optimal input. 

## 1. Getting Started

### 1.1. Install dependencies

Create a **Python 3.11.11** virtual environment:

```bash
python3.11.11 -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 1.2. Setup `.env` file

Provide an `OPENAI_API_KEY` in the `.env` file:

```.env
OPENAI_API_KEY = "Your API Key here"
```

## 2. Pipeline:

### 2.1. Parse the PDF into a Structured Format (HTML/Tabular/Markdown).
- [x] Image Preprocessing.
- [x] Define parsing logics.
- [ ] Define a generalized structured format.
- [ ] Enable multi-document parsing support.
- [ ] Perform sanity check on OpenAI outputs.
### 2.2. Define data visualization features.
- [ ] Define different graphs for efficient data visualization.
- [ ] Define methods for efficient data querying.
- [ ] **OPTIONAL**: Deploy simple ML/DL models for predicting data.
- [ ] Frontend & Backend development.
### 2.3. Integrate Agentic System for Analytics & Recommendation.
- [ ] Define Agent Architecture (Agentic/MAS).
- [ ] Build Agent Architecture.
- [ ] Finetune/Define strategies for Agents to analyze Financial Reports.