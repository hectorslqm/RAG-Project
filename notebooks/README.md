# RAG Coursework — How to Run

Evaluation of a Retrieval-Augmented Generation (RAG) system on SQuAD v2,
comparing two LLM providers across three prompting configurations.

## Contents

| File / folder          | Description                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `coursework_rag.ipynb` | Main notebook: code, documentation, figures and analysis.                            |
| `results/`             | Experiment output (`results_seed42/43/44.jsonl`). Required by the analysis sections. |
| `figures/`             | Experiment generated figures.                                                        |
| `requirements.txt`     | Python dependencies.                                                                 |

The SQuAD v2 dataset is downloaded automatically by the notebook, so it is not
included here.

## 1. Requirements

Python 3.12 or newer.

## 2. Install the libraries

From this folder, create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Configure the API keys

The notebook calls two hosted LLM providers, so it needs two API keys. Create a
file named `.env` in this same folder with the following content, replacing the
placeholders with your own keys:

```
OPENAI_API_KEY=your_openai_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here
```

The notebook loads this file automatically with `python-dotenv`. The `.env` file
is not included in this submission because it contains private credentials.

## 4. Run the notebook

```bash
jupyter notebook coursework_rag.ipynb
```

Then run the cells in order.

## Reproducing the analysis without calling the APIs

Section 6 runs the full experiment and makes paid API calls. To skip it and
reproduce the reported results directly from the saved output, keep
`LOAD_FROM_FILE = True` in Section 6.1 and make sure the `results/` folder sits
next to the notebook. With that flag on, every section from 7 onwards
(tables and figures) runs offline, and no API keys are needed.
