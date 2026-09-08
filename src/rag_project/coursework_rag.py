# %% [markdown]
# # Evaluation of a Retrieval-Augmented Generation (RAG) system using "gpt-5.4-mini" and "muse-glimmer-30b" models on SQuAD V2 Dataset. Principles of Machine Learning (7WCM2032) Coursework
#
# **Student Name**: Hector S. Lazcano Quintero Marmol  
# **Student ID**: 25054284
#
# # Copyright and Licensing
#
# ##
#
# This Jupyter notebook is provided by Hector S. Lazcano Quintero Marmol for educational purposes. You are free to use, share, and modify the contents of this notebook under the following conditions:
#
# - **Attribution**: You must give appropriate credit, provide a link to the original source, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the author or the University of Hertfordshire endorses you or your use.
#
#   **Suggested Attribution**:
#
#   This notebook was originally created by Hector Samuel Lazcano Quintero Marmol for the Principles of Machine Learning Module (7WCM2032) using as a starting point of reference the `Unit4_LLMs-1.ipynb`, Principles of Machine Learning (7WCM2032), University of Hertfordshire, 2024. Providing the correct attributions.
#
# ## Course material — Manal Helal, University of Hertfordshire
#
# © 2024 Manal Helal, University of Hertfordshire
#
# This Jupyter notebook is provided by Manal Helal, a lecturer at the University of Hertfordshire, for educational purposes. You are free to use, share, and modify the contents of this notebook under the following conditions:
#
# 1. **Attribution**: You must give appropriate credit, provide a link to the original source, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the lecturer or the University of Hertfordshire endorses you or your use.
#
#    **Suggested Attribution**:
#
#    This notebook was originally created by Manal Helal, University of Hertfordshire for the Principles of Machine Learning Module (7WCM2032) 2024.
#
# 2. **No Warranty**: The content of this notebook is provided "as-is," without warranty of any kind. The lecturer and the University of Hertfordshire make no representations or warranties, either express or implied, as to the accuracy, reliability, or completeness of the information provided herein.
#
# 3. **Limited Liability**: In no event shall the lecturer or the University of Hertfordshire be liable for any damages arising from the use of, or inability to use, the contents of this notebook, including but not limited to damages for loss of data, loss of profits, or interruption of business, even if advised of the possibility of such damages.
#
# 4. **Environment and Compatibility**: This notebook has been developed and tested in a specific environment. The lecturer and the University of Hertfordshire cannot guarantee that the notebook will function as expected in different environments. Users are responsible for ensuring compatibility and for addressing any issues that may arise.
#
# For any questions or further information, please contact Manal Helal at m.helal@herts.ac.uk
#
# ### Attribution and record of changes
#
# Provided in satisfaction of condition 1 above.
#
# **Original source**: `Unit4_LLMs-1.ipynb`, Principles of Machine Learning (7WCM2032), University of Hertfordshire, 2024.
#
# **Adapted from the original source, with modifications:**
#
# | Component              | Modification                                                                                                                                                                                               |
# | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | `EmbeddingModel`       | `embed_single` merged into a single overloaded `embed` accepting either a string or a list; type aliases added.                                                                                            |
# | `SimpleVectorDatabase` | Metadata ids assigned automatically using an offset from the insertion point to prevent retrieved documents from being untraceable due to mismatched ids.                                                  |
# | `SimpleRAGSystem`      | Prompt template rewritten to require short answers to reduce the consumption of tokens, and reduce the refusal phrases to one sentence.                                                                    |
# | `RAGResponse`          | Extended to include LLMResults, this container carries the token usage and latency from the LLM call.                                                                                                      |
# | `RAGEvaluator`         | The original substring support check was replaced with the official SQuAD v2 protocol: exact match and token-level F1, maximised over reference answers, with decline handling for unanswerable questions. |
#
# **Written for this coursework (not derived from the original source):** the `LLM` provider registry with the `OpenAILLM` and `NVidiaLLM` implementations, `LLMResults`, the `Dataset` and `DatasetQuestion` wrappers over SQuAD v2, the `NoRAG` baseline, the `Experiment` and `ExperimentResult` harness, and the whole of Sections 7 and 8 — the analysis DataFrame, the `Plot` class and every summary table.
#
# ## Dataset — SQuAD v2
#
# The Stanford Question Answering Dataset v2.0 is distributed under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** licence.
# Dataset card: https://huggingface.co/datasets/rajpurkar/squad_v2
#
# ## Retrieval model — all-MiniLM-L6-v2
#
# Sentence embeddings are produced with `sentence-transformers/all-MiniLM-L6-v2`, released under the **Apache License 2.0**. The model maps text to a 384-dimensional dense vector space and is used here to embed both the corpus paragraphs and the questions.
# Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#
# ## Models evaluated
#
# Neither model is redistributed in this work; both were accessed over their providers' APIs and are subject to those providers' terms of use.
#
# | Model                   | Accessed through                                   |
# | ----------------------- | -------------------------------------------------- |
# | `gpt-5.4-mini`          | OpenAI API                                         |
# | `meta/muse-glimmer-30b` | NVIDIA API (`https://integrate.api.nvidia.com/v1`) |
#
# ## Software
#
# | Library                 | Version | Licence                  |
# | ----------------------- | ------- | ------------------------ |
# | `sentence-transformers` | 6.0.0   | Apache 2.0               |
# | `datasets`              | 5.0.1   | Apache 2.0               |
# | `openai`                | 3.6.0   | Apache 2.0               |
# | `numpy`                 | 2.5.2   | BSD-3-Clause             |
# | `pandas`                | 3.0.5   | BSD-3-Clause             |
# | `matplotlib`            | 3.11.1  | PSF (matplotlib licence) |
# | `python-dotenv`         | 1.2.3   | BSD-3-Clause             |
#
# ---
#
# # References
#
# Referenced in Harvard style, ordered alphabetically by author. Organisations are
# cited as corporate authors and alphabetised under the organisation name.
#
# Helal, M. (2024) _Unit4_LLMs-1.ipynb_ [Jupyter notebook]. Principles of Machine Learning (7WCM2032). Hatfield: University of Hertfordshire.
#
# HuggingFace Evaluate Authors (2020) `squad_v2.py`: SQuAD v2 evaluation metric. Apache License 2.0. Available at: https://github.com/huggingface/evaluate/blob/main/metrics/squad_v2/squad_v2.py (Accessed: 7 September 2026).
#
# Rajpurkar, P. (2018) `_squad_v2_ [Dataset]`. Hugging Face. Available at: https://huggingface.co/datasets/rajpurkar/squad_v2 (Accessed: 7 September 2026).
#
# Rajpurkar, P., Jia, R. and Liang, P. (2018) 'Know what you don't know: unanswerable questions for SQuAD', _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_. Melbourne, Australia: Association for Computational Linguistics, pp. 784–789. Available at: https://doi.org/10.18653/v1/P18-2124 (Accessed: 7 September 2026).
#
# Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: sentence embeddings using Siamese BERT-networks', _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing_. Hong Kong: Association for Computational Linguistics. Available at: https://arxiv.org/abs/1908.10084 (Accessed: 7 September 2026).
#
# Sentence-Transformers (no date) _all-MiniLM-L6-v2_ [Machine learning model]. Hugging Face. Available at: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (Accessed: 7 September 2026).
#
# Waskom, M. (no date) _Visualizing distributions of data_. seaborn documentation. Available at: https://seaborn.pydata.org/tutorial/distributions.html (Accessed: 7 September 2026).
#
# ## BibTeX entries
#
# Provided for convenience only; the Harvard list above is the reference list.
#
# ```bibtex
# @inproceedings{rajpurkar-etal-2018-know,
#     title = "Know What You Don{'}t Know: Unanswerable Questions for {SQ}u{AD}",
#     author = "Rajpurkar, Pranav  and
#       Jia, Robin  and
#       Liang, Percy",
#     editor = "Gurevych, Iryna  and
#       Miyao, Yusuke",
#     booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
#     month = jul,
#     year = "2018",
#     address = "Melbourne, Australia",
#     publisher = "Association for Computational Linguistics",
#     url = "https://aclanthology.org/P18-2124",
#     doi = "10.18653/v1/P18-2124",
#     pages = "784--789",
#     eprint={1806.03822},
#     archivePrefix={arXiv},
#     primaryClass={cs.CL}
# }
# ```
#
# ```bibtex
# @inproceedings{reimers-2019-sentence-bert,
#     title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
#     author = "Reimers, Nils and Gurevych, Iryna",
#     booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
#     month = "11",
#     year = "2019",
#     publisher = "Association for Computational Linguistics",
#     url = "https://arxiv.org/abs/1908.10084"
# }
# ```
#
# ---
#
# # Declaration of Use of Generative AI
#
# > **DRAFT — check this against the module's own policy on generative AI before submitting, and edit or remove it accordingly.**
#
# Generative AI was used as a coding and analysis assistant during the preparation of this notebook:
#
# - **Claude (Anthropic)** was used in Sections 7 and 8. It contributed to the implementation of the `hydrate` helper, the empirical cumulative distribution and token-cost figures, the grid layout of the similarity histograms, and the `threshold_table` and `cost_table` summary tables. It was also used to discuss the interpretation of the results.
# - **GitHub Copilot** was used throughout for small in-editor corrections and completions while writing the code.
#
# The following tools are declared for completeness, although they are not generative AI. **ruff**, a deterministic static linter and formatter, was used to check style and formatting across the project. **Pylance** was used in the editor for type checking and diagnostics. Both analyse code against fixed rules and type information; neither generates or suggests new logic.
#
# All experimental design decisions, the choice of research questions, the execution of the experiment and the conclusions drawn in the written report are the author's own. Every figure and table reported here was regenerated from the raw experimental records in `results/` and checked against them.
#

# %% [markdown]
# # Index
#
# - **Section 1**: EMBEDDINGS AND VECTOR SEARCH
# - **Section 2**: LLM Models with two providers + SQuAD Dataset
#   - **Section 2.1**: SQuAD Dataset
# - **Section 3**: RETRIEVAL-AUGMENTED GENERATION (RAG)
# - **Section 4**: Evaluation
# - **Section 5**: Build Experiment
# - **Section 6**: Run the experiment
#   - **Section 6.1**: Load results for post-processing and analysis
# - **Section 7**: Post-processing and analysis
#   - **Section 7.1**: Build the analysis DataFrame
#   - **Section 7.2**: Plotting functions for analysis
#   - **Section 7.3**: Summary tables for analysis
# - **Section 8**: Run the analysis and generate plots and tables
#   - **Section 8.1**: Summary of every metric in the brief
#   - **Section 8.2**: Where the grounded answers went wrong
#   - **Section 8.3**: Figures
#   - **Section 8.4**:
#   - **Section 8.5**:
#   - **Section 8.6**:
#   - **Section 8.7**:
#

# %% [markdown]
# ## SECTION 1: EMBEDDINGS AND VECTOR SEARCH
#

# %%
import time
from dataclasses import dataclass, field
from typing import overload

import numpy as np

type Vector = np.ndarray  # forma (dim,)
type Matrix = np.ndarray  # forma (n, dim)


class EmbeddingModel:
    """Wrapper for generating text embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model.

        Args:
            model_name: Name of the sentence-transformers model
        """
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_embedding_dimension()

    @overload
    def embed(self, texts: str) -> Vector: ...

    @overload
    def embed(self, texts: list[str]) -> Matrix: ...

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Generate embeddings for a text or list of texts.

        Args:
            texts: A text string or list of text strings to embed

        Returns:
            numpy array of shape (len(texts), dimension) if texts is a list, or (dimension,) if texts is a single string
        """
        if isinstance(texts, str):
            # Generate embedding for a single text.
            return self._model.encode([texts], normalize_embeddings=True)[0]

        return self._model.encode(texts, normalize_embeddings=True)


class SimpleVectorDatabase:
    """In-memory vector database for storing and retrieving embeddings."""

    def __init__(self, embedding_model: EmbeddingModel):
        self._embedding_model = embedding_model
        self._documents: list[str] = []
        self._metadata: list[dict] = []
        self._embeddings: np.ndarray | None = None

    def add_documents(
        self, documents: list[str], metadata: list[dict] | None = None
    ) -> None:
        """Add documents to the vector database.

        Args:
            documents: List of text documents
            metadata: Optional metadata for each document
        """
        if metadata is None:
            # Offset by the current index size to ensure unique ids.
            start = len(self._documents)
            metadata = [{"id": start + i} for i in range(len(documents))]

        # Generate embeddings for all documents at once
        embeddings = self._embedding_model.embed(documents)

        self._documents.extend(documents)
        self._metadata.extend(metadata)

        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

    def search(self, query: str, top_k: int = 4) -> list[tuple[str, float, dict]]:
        """Search for the most similar documents to a query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of tuples (document_text, similarity_score, metadata)
        """
        query_vec = self._embedding_model.embed(query)

        # Compute cosine similarity
        similarities = np.dot(self._embeddings, query_vec)  # type: ignore

        # Get top_k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                (self._documents[idx], float(similarities[idx]), self._metadata[idx])
            )

        return results


# %% [markdown]
# ## SECTION 2: LLM Models with two providers + SQuAD Dataset
#

# %%
import os
from typing import ClassVar

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class LLMResults:
    """Container for the results returned by the LLM."""

    answer: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


class LLM:
    """
    Creates a base class for Large Language Models (LLMs) with a registry for different providers.

    Raises:
        NotImplementedError: Raised when the generate method is not implemented in the subclass.
        ValueErrorption: Raised when an unsupported model provider is specified.

    Returns:
        An instance of the LLM subclass corresponding to the specified provider.
    """

    provider: ClassVar[str]
    _registry: ClassVar[dict[str, type["LLM"]]] = {}

    @property
    def model_name(self) -> str:
        """Return the name of the model being used."""
        return ""

    def generate(self, prompt: str) -> LLMResults:
        """Generates a response from the language model based on the given prompt.

        Args:
            prompt (str): The input prompt to generate a response for.

        Raises:
            NotImplementedError: Raised when the generate method is not implemented in the subclass.

        Returns:
            An instance of LLMResults containing the generated response and associated metadata.
        """
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        """
        Registers the subclass in the LLM registry based on its provider attribute.
        """
        super().__init_subclass__(**kwargs)
        if provider := getattr(cls, "provider", None):
            LLM._registry[provider.upper()] = cls

    @staticmethod
    def create(provider: str, model_name: str) -> "LLM":
        """
        Creates an instance of the LLM subclass corresponding to the specified provider.

        Args:
            provider (str): The name of the model provider (e.g., "OPENAI", "NVIDIA").
            model_name (str): The name of the model to be used.

        Raises:
            ValueError: Raised when an unsupported model provider is specified.

        Returns:
            An instance of the LLM subclass corresponding to the specified provider.
        """
        try:
            factory = LLM._registry[provider.upper()]
            return factory(model_name)  # type: ignore[call-arg]
        except KeyError:
            raise ValueError(f"Unsupported model provider: {provider}")


class OpenAILLM(LLM):
    """
    A subclass of LLM to interact specifically with OpenAI's language models.
    **IMPORTANT**:
        Make sure to set the OPENAI_API_KEY environment variable before using this class.
    Args:
        model_name (str): The name of the OpenAI model to be used.

    Returns:
        OpenAILLM: An instance of the OpenAILLM class initialized with the specified model.
    """

    # Define the provider for the OpenAILLM class
    provider: ClassVar[str] = "OPENAI"

    API_KEY = os.getenv("OPENAI_API_KEY")

    def __init__(self, model_name: str):
        self._client = OpenAI(api_key=self.API_KEY)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Return the name of the model being used."""
        return self._model_name

    def generate(self, prompt: str) -> LLMResults:
        start = time.perf_counter()
        response = self._client.responses.create(
            model=self._model_name,
            input=prompt,
            reasoning={"effort": "medium"},
        )
        latency = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        return LLMResults(
            answer=response.output_text,
            latency_s=latency,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
        )


class NVidiaLLM(LLM):
    """
    A subclass of LLM to interact specifically with NVIDIA's language models.
    **IMPORTANT**:
        Make sure to set the NVIDIA_API_KEY environment variable before using this class.

    Args:
        model_name (str): The name of the NVIDIA model to be used.

    Returns:
        NVidiaLLM: An instance of the NVidiaLLM class initialized with the specified model.
    """

    # Define the provider for the NVidiaLLM class
    provider: ClassVar[str] = "NVIDIA"

    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    API_KEY = os.getenv("NVIDIA_API_KEY")

    def __init__(self, model_name: str):
        self._client = OpenAI(base_url=self.NVIDIA_BASE_URL, api_key=self.API_KEY)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Return the name of the model being used."""
        return self._model_name

    def generate(self, prompt: str) -> LLMResults:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            # Limiting the range of randomness to make the output reproducible
            temperature=0,
            top_p=0.95,
            max_tokens=8192,
            stream=False,
        )
        latency = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        return LLMResults(
            answer=response.choices[0].message.content or "",
            latency_s=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


# %% [markdown]
# ### SECTION 2.1: SQuAD Dataset
#

# %%
import random

from datasets import load_dataset


@dataclass
class DatasetQuestion:
    """Container for a single question in the dataset."""

    id: str
    question: str
    # If empty, the question is unanswerable
    answers: list[str]
    # Index of the correct paragraph within the corpus
    context_id: int
    is_answerable: bool = field(init=False)

    # Assign is_answerable based on whether there are any answers. Right after initialization
    def __post_init__(self):
        self.is_answerable = len(self.answers) > 0


class Dataset:
    """Dataset wrapper for loading contexts and questions.
        - SQuAD v2 dataset by default. But can be configured to use other datasets as well.
            - This dataset is composed by two subsets: "train" and "validation". We are using the "validation" subset by default.
        - Seed for random number generation to ensure reproducibility. 42 by default.

        See https://huggingface.co/datasets/rajpurkar/squad_v2 to explore the dataset.

        @inproceedings{rajpurkar-etal-2018-know,
        title = "Know What You Don{'}t Know: Unanswerable Questions for {SQ}u{AD}",
        author = "Rajpurkar, Pranav  and
          Jia, Robin  and
          Liang, Percy",
        editor = "Gurevych, Iryna  and
          Miyao, Yusuke",
        booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
        month = jul,
        year = "2018",
        address = "Melbourne, Australia",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/P18-2124",
        doi = "10.18653/v1/P18-2124",
        pages = "784--789",
        eprint={1806.03822},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    """

    def __init__(
        self,
        dataset: str = "rajpurkar/squad_v2",
        split: str = "validation",
        seed: int = 42,
    ):
        CONTEXT = "context"
        QUESTION = "question"
        ANSWERS = "answers"
        TEXT = "text"

        # Load the specified split of the dataset and initialize the random number generator.
        self._dataset_name = dataset
        self._split = split
        self._seed = seed
        self._dataset = load_dataset(dataset, split=split)
        self._random = random.Random(seed)
        # Corpus of unique contexts sorted in ascending order
        self._corpus = sorted(self._dataset.unique(CONTEXT))
        # Mapping from context to its index in the corpus
        self._context_to_id = {context: idx for idx, context in enumerate(self._corpus)}

        self._questions: list[DatasetQuestion] = []
        self._questions_by_id: dict[str, DatasetQuestion] = {}
        # Create a QAPair for each row in the dataset and store it
        for row in self._dataset:
            question = DatasetQuestion(
                id=row["id"],  # type: ignore
                question=row[QUESTION],  # type: ignore
                # Remove duplicated answers while preserving order
                answers=list(dict.fromkeys(row[ANSWERS][TEXT])),  # type: ignore
                # Store the corresponding id of the context in the corpus
                context_id=self._context_to_id[row[CONTEXT]],  # type: ignore
            )
            self._questions.append(question)
            self._questions_by_id[question.id] = question

    @property
    def dataset_name(self) -> str:
        """Return the name of the dataset being used."""
        return self._dataset_name

    @property
    def split(self) -> str:
        """Return the split of the dataset being used."""
        return self._split

    @property
    def seed(self) -> int:
        """Return the seed used for random number generation."""
        return self._seed

    @property
    def corpus(self) -> list[str]:
        """Return the list of unique contexts in the corpus."""
        return self._corpus

    def get_questions(
        self, n: int | None = None, only_answerable: bool | None = None
    ) -> list[DatasetQuestion]:
        """
        Return a list of questions based on the specified criteria.

        n: Number of questions to return. If None (or larger than the pool), return all matching questions in dataset order.
        only_answerable: If True, return only answerable questions. If False, only unanswerable ones. If None, no filtering.

        Returns:
            A list of Question objects. When n is smaller than the pool, a random
            sample drawn from the generator, so results are reproducible.
        """
        questions = self._questions
        if only_answerable is not None:
            questions = [q for q in questions if q.is_answerable == only_answerable]
        if n is None or n >= len(questions):
            return list(questions)
        return self._random.sample(questions, n)

    def _get_question_by_id(self, question_id: str) -> DatasetQuestion:
        question = self._questions_by_id.get(question_id)
        if question is None:
            raise ValueError(f"Question with ID {question_id} not found.")
        return question

    def context_for_question(self, question_id: str) -> str:
        """Return the context corresponding to the given question ID. This will be used to measure retrieval accuracy."""
        return self._corpus[self.get_context_id(question_id)]

    def get_context_id(self, question_id: str) -> int:
        return self._get_question_by_id(question_id).context_id

    def __len__(self) -> int:
        """Return the number of questions in the dataset."""
        return len(self._questions)

    def statistics(self) -> dict[str, int]:
        """Return basic statistics about the dataset."""
        answerable_questions = sum(1 for q in self._questions if q.is_answerable)
        return {
            "num_questions": len(self._questions),
            "num_contexts": len(self._corpus),
            "num_answerable_questions": answerable_questions,
            "num_unanswerable_questions": len(self._questions) - answerable_questions,
        }


# %% [markdown]
# ## SECTION 3: RETRIEVAL-AUGMENTED GENERATION (RAG)
#

# %%
@dataclass
class RAGResponse:
    """Container for RAG response with sources."""

    answer: str
    sources: list[dict]
    context_used: list[str]
    confidence: float
    llm_results: LLMResults | None


UNANSWERABLE_PHRASE = "I don't have an answer"


class SimpleRAGSystem:
    """
    A complete RAG system that:
    1. Indexes documents
    2. Retrieves relevant context
    3. Generates grounded answers
    """

    def __init__(self, vector_db: SimpleVectorDatabase, llm_model: LLM):
        self._vector_db = vector_db
        self._llm_model = llm_model

    def _format_context(self, retrieved_docs: list[tuple[str, float, dict]]) -> str:
        """Format retrieved documents for inclusion in the prompt."""
        formatted = []
        for i, (doc, score, meta) in enumerate(retrieved_docs, 1):
            formatted.append(f"[{i}] {doc}")
        return "\n\n".join(formatted)

    def _create_prompt(
        self, question: str, context: list[tuple[str, float, dict]]
    ) -> str:
        """Create the complete prompt."""
        context_str = self._format_context(context)

        prompt = """Answer using ONLY the context below. I want you to answer in the shortest possible way. If the answer is not present, say exactly: "{unanswerable_phrase}"

Context:
{context}

Question: {question}

Answer:"""

        return prompt.format(
            context=context_str,
            question=question,
            unanswerable_phrase=UNANSWERABLE_PHRASE,
        )

    def answer(self, question: str, top_k: int = 4) -> RAGResponse:
        """Generate a RAG answer for a question."""
        answer, llm_results = None, None
        # Retrieve relevant documents
        retrieved = self._vector_db.search(question, top_k=top_k)

        # Create the prompt
        prompt = self._create_prompt(question, retrieved)

        # Generate answer using the LLM
        if self._llm_model is not None:
            llm_results = self._llm_model.generate(prompt)
            answer = llm_results.answer
        # If no LLM model is provided, the answer will remain None.

        # Extract sources for transparency
        sources = [
            {
                # Truncate the document text to 100 characters
                "text": doc[:100] + "..." if len(doc) > 100 else doc,
                "score": score,
                "metadata": meta,
            }
            for doc, score, meta in retrieved
        ]

        return RAGResponse(
            answer=answer or "",
            sources=sources,
            context_used=[doc for doc, _, _ in retrieved],
            confidence=retrieved[0][1] if retrieved else 0.0,
            llm_results=llm_results,
        )


class NoRAG:
    """
    This class provides an answer without using the RAG system.
    """

    def __init__(self, llm_model: LLM):
        self._llm_model = llm_model

    def _create_prompt(
        self, question: str, context: list[tuple[str, float, dict]] | None = None
    ) -> str:
        """Create the complete prompt."""
        prompt = """I want you to answer in the shortest possible way. If you do not know the answer, say exactly: "{unanswerable_phrase}"

Question: {question}

Answer:"""

        return prompt.format(question=question, unanswerable_phrase=UNANSWERABLE_PHRASE)

    def answer(self, question: str) -> RAGResponse:
        """Get a plain answer for a question without additional context."""
        prompt = self._create_prompt(question=question)
        llm_results = self._llm_model.generate(prompt)
        return RAGResponse(
            answer=llm_results.answer,
            sources=[],
            context_used=[],
            confidence=0.0,
            llm_results=llm_results,
        )


# %% [markdown]
# ## SECTION 4: Evaluation
#

# %%
import re
import string
from collections import Counter


def _normalize_texts(text: str) -> str:
    """Lowercase, strip punctuation and articles, and normalize whitespace.
    E.G. "The CAR runs fast." -> "car runs fast"
    The normalization follows the SQuAD v2 evaluation protocol for exact match and F1 score calculations. See the Copyright and Licensing section of the notebook for the source of this normalization function.
    """
    text = text.lower().strip()
    text = "".join(
        character for character in text if character not in string.punctuation
    )

    # Remove articles (a, an, the) from the text.
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    return " ".join(text.split())


# Normalized should look like: i dont have answer
UNANSWERABLE_TOKEN = _normalize_texts(UNANSWERABLE_PHRASE)


def clean_model_answer(raw: str | None) -> str:
    """Clean the model's raw answer.

    Args:
        raw (str | None): The raw answer from the model.

    Returns:
        str: The cleaned answer, with leading/trailing whitespace removed. Returns an empty string if the input is None.
    """
    if raw is None:
        return ""
    return raw.strip()


class RAGEvaluator:
    """Evaluate a RAG system against test questions.

    Adapted from the module notebook's RAGEvaluator: the substring-support
    check was replaced with the official SQuAD v2 protocol (EM / token-F1,
    max over reference answers, decline handling for unanswerable questions).

    """

    @staticmethod
    def is_declined(answer: str) -> bool:
        """True if the model used the UNANSWERABLE_PHRASE"""
        return UNANSWERABLE_TOKEN in _normalize_texts(answer)

    @staticmethod
    def _em_single(prediction: str, reference: str) -> float:
        """Exact match between one prediction and one reference answer"""
        return float(_normalize_texts(prediction) == _normalize_texts(reference))

    @staticmethod
    def _f1_single(prediction: str, reference: str) -> float:
        """Token-overlap F1 between one prediction and one reference answer.
        source: https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/
        """
        prediction_tokens = _normalize_texts(prediction).split()
        reference_tokens = _normalize_texts(reference).split()
        if not prediction_tokens or not reference_tokens:
            return float(prediction_tokens == reference_tokens)
        # Compute the number of overlapping tokens between prediction and reference
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        # Compute the overlap and calculate precision, recall, and F1 score
        overlap = sum(common.values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        # Compute the F1 Score
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def evaluate_answer(cls, prediction: str, reference_answers: list[str]) -> dict:
        """Evaluate a single answer against the reference answers.
        Returns:
            A dictionary with keys "em", "f1", and "declined". "em" and "f1" are float scores, "declined" is a boolean indicating if the model declined to answer.
        """
        prediction = clean_model_answer(prediction)
        declined = cls.is_declined(prediction)

        if not reference_answers:  # unanswerable question
            score = float(declined)
            return {"em": score, "f1": score, "declined": declined}

        if declined:  # declined an answerable question
            return {"em": 0.0, "f1": 0.0, "declined": True}

        em = max(cls._em_single(prediction, ref) for ref in reference_answers)
        f1 = max(cls._f1_single(prediction, ref) for ref in reference_answers)
        return {"em": em, "f1": f1, "declined": False}

    @classmethod
    def is_supported_by_sources(
        cls, prediction: str, used_contexts: list[str]
    ) -> bool | None:
        """Answer-supported-by-source rat.
        Returns:
            True if the normalized answer appears inside any context used, False if not. None when there is nothing to check (declined answers or plain runs with no RAG)
        """

        prediction = clean_model_answer(prediction)
        if not used_contexts or cls.is_declined(prediction):
            return None
        normalized_prediction = _normalize_texts(prediction)
        if not normalized_prediction:
            return None

        # Compare whole words, not raw characters. A plain substring test
        # matches inside longer words ("no" is found in "north"), which would
        # inflate this rate: about 11% of SQuAD v2 answers are 4 characters
        # or shorter, so the spurious matches are not a rare edge case.
        prediction_tokens = normalized_prediction.split()

        def _contains(context: str) -> bool:
            context_tokens = _normalize_texts(context).split()
            n = len(prediction_tokens)
            return any(
                context_tokens[i : i + n] == prediction_tokens
                for i in range(len(context_tokens) - n + 1)
            )

        return any(_contains(context) for context in used_contexts)


# %% [markdown]
# ## SECTION 5: Build Experiment
#

# %%
import json
from dataclasses import asdict
from pathlib import Path


@dataclass
class ExperimentResult:
    """Container for a single experiment result."""

    llm_model: str
    config: str
    seed: int
    question_id: str
    is_answerable: bool
    answer: str
    em: float
    f1: float
    declined: bool
    retrieval_hit: bool | None
    supported_by_source: bool | None
    retrieved_ids: list[int]
    top_similarity: float
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


class Experiment:
    """Package the experiment configuration and results in a single object."""

    def __init__(self):
        self._embedding_model = EmbeddingModel()
        self._records: list[ExperimentResult] = []

    def _read_processed_pairs(self, results_file: Path) -> set[tuple[str, str, str]]:
        """Return a list of tuples (llm_model, config, question_id) from the processed results"""
        done = set()
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    record = json.loads(line)
                    done.add(
                        (record["llm_model"], record["config"], record["question_id"])
                    )
        return done

    @property
    def records(self) -> list[ExperimentResult]:
        """Return the list of experiment records."""
        return self._records

    def clear_records(self, results_file: Path, seed: int):
        """Clear records for this seed and remove its results file if it exists."""
        self._records = [r for r in self._records if r.seed != seed]
        if results_file.exists():
            results_file.unlink()

    def run_experiment(
        self,
        dataset: Dataset,
        llms: list[LLM],
        seed: int,
        n_questions: int,
        top_k_values: list[int | None],
        results_path: Path = Path("results"),
        replace_existing: bool = False,
    ):
        """Run the experiment with the given configuration and save results to a file.
        dataset: Dataset object containing the questions and contexts
        llms: List of LLM objects to be evaluated
        seed: Random seed for reproducibility
        n_questions: Number of questions to sample from the dataset for evaluation
        top_k_values: List of top_k values for retrieval. Use None to include the NoRAG baseline in the experiment.
        results_path: Path to the directory where results will be saved. Defaults to "results".
        replace_existing: If True, clear existing records for this seed and remove the results file if it exists.
        """
        results_path.mkdir(exist_ok=True)
        # Initialize the vector database, and add documents from the dataset's corpus. with the given seed
        vector_db = SimpleVectorDatabase(self._embedding_model)
        vector_db.add_documents(dataset.corpus)
        # Retrieve a random sample of questions from the dataset.
        questions = dataset.get_questions(n=n_questions)
        # Create the results file path and check for already completed evaluations to avoid redundant computations.
        results_file = results_path / f"results_seed{seed}.jsonl"

        # If replace_existing is True, clear existing records for this seed and remove the results file if it exists.
        # Otherwise, read already processed pairs from the results file to skip them during evaluation.
        if replace_existing:
            self.clear_records(results_file, seed)
            processed = set()
        else:
            processed = self._read_processed_pairs(results_file)

        print("=" * 60)
        print(f"Running experiment with seed {seed}, {len(questions)} questions.")
        print(f"Dataset: {dataset.dataset_name}, Split: {dataset.split}, Seed: {seed}")
        with open(results_file, "a") as f:
            for top_k in top_k_values:
                for llm in llms:
                    print(
                        f"RAG With Top-k retrieval: {top_k if top_k is not None else 'Plain Prompting (No RAG)'}"
                    )
                    print(f"LLM model: {llm.model_name}")

                    model_name = llm.model_name
                    # Initialize the system. If top_k is None, use NoRAG; otherwise, use SimpleRAGSystem with the specified top_k.
                    if top_k is not None:
                        config = f"rag_k-{top_k}"
                        # Initialize the RAG System with the current LLM model and vector database
                        system = SimpleRAGSystem(vector_db, llm_model=llm)
                    else:
                        config = "plain"
                        system = NoRAG(llm_model=llm)

                    for i, question in enumerate(questions, start=1):
                        # Check if the current configuration and question ID have already been evaluated to avoid redundant computations.
                        if (model_name, config, question.id) in processed:
                            continue

                        if isinstance(system, SimpleRAGSystem) and top_k is not None:
                            response = system.answer(question.question, top_k=top_k)
                            retrieved_ids = [
                                source["metadata"]["id"] for source in response.sources
                            ]

                            retrieval_hit = question.context_id in retrieved_ids
                        else:
                            response = system.answer(question.question)
                            retrieved_ids = [
                                source["metadata"]["id"] for source in response.sources
                            ]

                            retrieval_hit = None

                        answer = clean_model_answer(response.answer)
                        metrics = RAGEvaluator.evaluate_answer(answer, question.answers)

                        supported = RAGEvaluator.is_supported_by_sources(
                            response.answer, response.context_used
                        )
                        if response.llm_results:
                            latency_s = response.llm_results.latency_s
                            prompt_tokens = response.llm_results.prompt_tokens
                            completion_tokens = response.llm_results.completion_tokens
                        else:
                            latency_s = 0.0
                            prompt_tokens = 0
                            completion_tokens = 0

                        record = ExperimentResult(
                            llm_model=model_name,
                            config=config,
                            seed=seed,
                            question_id=question.id,
                            is_answerable=question.is_answerable,
                            answer=answer,
                            em=metrics["em"],
                            f1=metrics["f1"],
                            declined=metrics["declined"],
                            retrieval_hit=retrieval_hit,
                            supported_by_source=supported,
                            retrieved_ids=retrieved_ids,
                            top_similarity=response.confidence,  # raw cosine sim
                            latency_s=latency_s,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                        self._records.append(record)
                        f.write(json.dumps(asdict(record)) + "\n")
                        f.flush()  # persist immediately
                        # Print progress and metrics for the current question, including the exact match (em), F1 score (f1), retrieval hit status, latency, and a truncated version of the answer.
                        print(
                            f"  [{i:>3}/{len(questions)}] em={metrics['em']:.0f} "
                            f"f1={metrics['f1']:.2f} hit={retrieval_hit} latency={latency_s:>5.1f}s | {answer[:100]}."
                        )
                    print(f"{config}: Processed")


# %% [markdown]
# ## SECTION 6: Run the experiment
#

# %%
# Initialize a LLM factory
llm = LLM()
## Create instances of the LLMs to be used in the experiment
muse_glimmer = LLM.create("NVIDIA", "meta/muse-glimmer-30b")
gpt_mini = LLM.create("OPENAI", "gpt-5.4-mini")
QUESTIONS_PER_SEED = 100
SEEDS = (42, 43, 44)
experiment = Experiment()
for seed in SEEDS:
    dataset = Dataset(dataset="rajpurkar/squad_v2", split="validation", seed=seed)
    experiment.run_experiment(
        dataset=dataset,
        llms=[gpt_mini, muse_glimmer],
        seed=seed,
        n_questions=QUESTIONS_PER_SEED,
        top_k_values=[None, 1, 5, 10],
        results_path=Path("results"),
        replace_existing=True,
    )


# %% [markdown]
# ### Section 6.1: Load results for post-processing and analysis
#

# %%
def load_results(results_dir: Path) -> list[ExperimentResult]:
    """Read every seed file into one flat list of records."""
    records = []
    for path in sorted(results_dir.glob("results_seed*.jsonl")):
        with open(path) as f:
            records.extend(
                ExperimentResult(**json.loads(line)) for line in f if line.strip()
            )
    return records


LOAD_FROM_FILE = True  # Set to True to load results from file instead of using the in-memory records from the experiment run above.
if LOAD_FROM_FILE and Path("results").exists():
    results = load_results(Path("results"))
    print(f"Loaded {len(results)} records from results directory.")
else:
    results = experiment.records
    print(f"Using {len(results)} records from the in-memory experiment run.")

# %% [markdown]
# ## SECTION 7: Post-processing and Analysis
#

# %% [markdown]
# ### SECTION 7.1: Build the analysis DataFrame
#

# %%
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Answers scoring at or above this F1 count as "correct" wherever the analysis
# needs a yes/no split (the grounding cross-tab and the similarity histogram).
# EM is too strict for that job: a right answer carrying one extra word scores
# EM = 0 but F1 ~ 0.9, and would be filed as a failure.
CORRECT_F1_THRESHOLD = 0.5


def results_to_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    """Turn the experiment records into the table the whole analysis reads from.

    Everything below this point becomes column arithmetic instead of nested
    loops, so each aggregation reads as the thing it computes.

    Three decisions are made here on purpose, once, instead of in every
    function downstream:

    - `retrieval_hit` and `supported_by_source` become pandas' nullable
      "boolean" dtype. In this experiment None means "not applicable" (the
      model declined, or it was a plain run with nothing retrieved) and NOT
      False. The nullable dtype keeps that difference, so `.mean()` skips
      those rows rather than counting them as zeros - which is exactly the
      mistake that would make a cautious model look good.
    - `config` becomes an *ordered* category, so every groupby, table and
      plot comes out as plain, rag_k-1, rag_k-5, rag_k-10 without anyone
      having to sort it again. Sorting the labels as text would put
      "rag_k-10" before "rag_k-5".
    - `correct`, `total_tokens` and `top_k` are derived once here.
    """
    frame = pd.DataFrame([asdict(r) for r in results])
    if frame.empty:
        return frame

    for column in ("retrieval_hit", "supported_by_source"):
        frame[column] = frame[column].astype("boolean")

    frame["correct"] = frame["f1"] >= CORRECT_F1_THRESHOLD
    frame["total_tokens"] = frame["prompt_tokens"] + frame["completion_tokens"]
    # "rag_k-10" -> 10. The plain baseline has no k, so it stays missing,
    # which is also what sorts it to the front below.
    frame["top_k"] = (
        frame["config"].str.extract(r"-(\d+)$", expand=False).astype("Int64")
    )

    ordered_configs = (
        frame[["config", "top_k"]]
        .drop_duplicates()
        .sort_values("top_k", na_position="first")["config"]
        .tolist()
    )
    frame["config"] = pd.Categorical(
        frame["config"], categories=ordered_configs, ordered=True
    )
    return frame


def config_order(frame: pd.DataFrame) -> list[str]:
    """Configurations present in the data, in reading order."""
    return list(frame["config"].cat.categories)


def model_order(frame: pd.DataFrame) -> list[str]:
    """Models present in the data, in a stable order."""
    return sorted(frame["llm_model"].unique())


def seed_order(frame: pd.DataFrame) -> list[int]:
    """Seeds present in the data."""
    return sorted(int(seed) for seed in frame["seed"].unique())


def top_k_values(frame: pd.DataFrame) -> list[int]:
    """The retrieval depths that were actually run."""
    return sorted(int(k) for k in frame["top_k"].dropna().unique())


def _mean_per_seed(
    frame: pd.DataFrame,
    model: str,
    config: str,
    metric: str,
    subset: bool | None = None,
) -> list[float]:
    """One mean per seed, for one model under one configuration.

    Each seed is a separate sample of 100 questions, so the seed is the unit
    we average and take the spread over. Averaging all 300 answers together
    would understate how much the score moves between samples.

    subset: True for answerable questions only, False for unanswerable only,
    None for both. Seeds where the metric is missing everywhere drop out
    instead of contributing a zero.
    """
    rows = frame[(frame["llm_model"] == model) & (frame["config"] == config)]
    if subset is not None:
        rows = rows[rows["is_answerable"] == subset]
    per_seed = rows.groupby("seed", observed=True)[metric].mean().dropna()
    return [float(value) for value in per_seed]


def hydrate(frame: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """Join the question text, the gold answers and the passages back onto the
    records, so a failure can be quoted in the report instead of only counted.

    The experiment stores ids, not prose. `retrieved_ids` are indices into
    `dataset.corpus`, which is `sorted(unique(context))` - a stable order that
    does not depend on the seed - so every passage is recoverable offline from
    the ids alone. Nothing here calls an API or re-runs the experiment.

    Any Dataset instance will do: the corpus and the id -> question lookup are
    seed-independent, only the *sample* of questions drawn from it is not.

    The retrieved passages are stored as references into the same corpus list,
    not as copies, so the extra columns cost list overhead rather than one
    duplicate of the text per row.
    """
    corpus = dataset.corpus
    questions = {question.id: question for question in dataset.get_questions()}

    return frame.assign(
        question=frame["question_id"].map(lambda qid: questions[qid].question),
        gold_answers=frame["question_id"].map(lambda qid: questions[qid].answers),
        gold_context=frame["question_id"].map(
            lambda qid: corpus[questions[qid].context_id]
        ),
        # A plain run retrieved nothing, so its list is empty rather than
        # missing: "the model was given no passages" is a fact, not a gap.
        retrieved_context=frame["retrieved_ids"].map(
            lambda ids: [corpus[doc_id] for doc_id in ids]
        ),
    )


def _largest_ecdf_gap(
    correct: list[float], incorrect: list[float]
) -> tuple[float, float]:
    """Biggest vertical distance between the two ECDFs, and where it happens.

    This is the Kolmogorov-Smirnov statistic, read here for what it means
    operationally rather than as a test: at the cut-off that maximises it, the
    gap is (share of wrong answers stopped) - (share of right answers lost).
    It is the ceiling on any single threshold on this axis, so a small number
    rules the whole idea out rather than just the cut-offs that were tried.
    """
    if not correct or not incorrect:
        return 0.0, float("nan")

    correct_sorted = np.sort(correct)
    incorrect_sorted = np.sort(incorrect)
    # Every observed value is a candidate cut-off; the extremum of a step
    # function can only sit at a step.
    cutoffs = np.unique(np.concatenate([correct_sorted, incorrect_sorted]))
    lost = np.searchsorted(correct_sorted, cutoffs, side="right") / len(correct_sorted)
    stopped = np.searchsorted(incorrect_sorted, cutoffs, side="right") / len(
        incorrect_sorted
    )
    gaps = stopped - lost
    best = int(np.argmax(gaps))
    return float(gaps[best]), float(cutoffs[best])


# %% [markdown]
# ### SECTION 7.2: Plotting functions for analysis
#

# %%
class Plot:
    """A class to encapsulate plotting functions for RAG experiment results."""

    # Define a set of colors for plotting different series in the analysis.
    # And keep consistent with the colors used in the notebook for visual clarity and comparison.
    SERIES_COLORS = ("#1b71af", "#ff0e62", "#0da70d", "#d1d408df", "#9467bd")

    def __init__(self, frame: pd.DataFrame, figure_path: Path = Path("figures")):
        self._frame = frame
        self._figure_path = figure_path

        # Update matplotlib's rcParams to customize the appearance of plots, including grid lines, colors, and other visual elements for better readability and aesthetics.
        plt.rcParams.update(
            {
                "axes.grid": True,
                "axes.grid.axis": "y",
                "grid.color": "#e5e4e0",
                "grid.linewidth": 0.8,
                "axes.axisbelow": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": "#b8b7b2",
                "axes.labelcolor": "#52514e",
                "text.color": "#0b0b0b",
                "xtick.color": "#52514e",
                "ytick.color": "#52514e",
                "figure.facecolor": "#fcfcfb",
                "axes.facecolor": "#fcfcfb",
            }
        )

    def plot_metric_by_config(
        self,
        frame: pd.DataFrame,
        metric: str = "f1",
        subset: bool | None = None,
        filename: str | None = None,
    ) -> None:
        """Grouped bars: one group per configuration, one bar per model.

        Error bars are the standard deviation across seeds, so they show how much
        the score moves when you resample the questions.
        """
        self._figure_path.mkdir(exist_ok=True)
        configs = config_order(frame)
        models = model_order(frame)

        subset_label = {
            None: "",
            True: " - answerable only",
            False: " - unanswerable only",
        }[subset]

        x = np.arange(len(configs))
        width = 0.8 / len(models)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        top = 0.0
        for i, model in enumerate(models):
            means, stds = [], []
            for config in configs:
                per_seed = _mean_per_seed(frame, model, config, metric, subset)
                means.append(float(np.mean(per_seed)) if per_seed else 0.0)
                stds.append(float(np.std(per_seed)) if per_seed else 0.0)

            # 2px gap between adjacent bars so the fills never touch
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                means,
                width * 0.92,
                yerr=stds,
                capsize=3,
                label=model,
                color=self.SERIES_COLORS[i % len(self.SERIES_COLORS)],
                error_kw={"ecolor": "#52514e", "elinewidth": 1},
            )
            # Direct labels: identity is never carried by colour alone. Sit them
            # above the error bar, not the bar, so the two never overlap.
            for bar, mean, std in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + std + 0.02,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#52514e",
                )
                top = max(top, mean + std)

        n_seeds = len(seed_order(frame))
        ax.set_xticks(x, configs)
        ax.set_ylabel(f"mean {metric.upper()}")
        # Leave room for the labels; never clip an error bar that runs past 1.0
        ax.set_ylim(0, max(1.05, top + 0.12))
        ax.set_title(
            f"{metric.upper()} by configuration{subset_label}\n"
            f"mean ± std across {n_seeds} seed(s)",
            fontsize=11,
            loc="left",
        )
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()

        name = filename or f"{metric}{subset_label.replace(' ', '_')}_by_config.png"
        fig.savefig(self._figure_path / name, dpi=150)
        plt.show()

    def plot_retrieval_hit_rate(self, frame: pd.DataFrame) -> None:
        """Line: share of questions whose own context made it into the top-k,
        split by whether the question has an answer at all.

        Retrieval runs before the LLM and uses the same embeddings for both
        models, so this is a property of the retriever alone - the models are
        not separated because they retrieved identical documents. Rows are
        de-duplicated by (seed, top_k, question) so a question is not counted
        once per model.

        Split because only the answerable line is a ceiling. Where an answer
        exists, a miss makes a correct answer impossible, so that line bounds
        everything downstream. An unanswerable question has nothing to find,
        so its hit-rate bounds nothing - it is shown because the gap is a
        finding of its own: SQuAD v2 writes unanswerable questions to look
        plausible against a paragraph, which makes them harder to retrieve.
        """
        self._figure_path.mkdir(exist_ok=True)
        rag = frame[frame["config"] != "plain"].dropna(subset=["retrieval_hit"])
        if rag.empty:
            print("No retrieval records to plot.")
            return

        per_question = rag.drop_duplicates(subset=["seed", "top_k", "question_id"])

        fig, ax = plt.subplots(figsize=(7, 4.5))
        # The two lines run close together at k=1, so their labels are pushed
        # to opposite sides of the marker instead of both sitting above it.
        series = [
            (
                True,
                "answerable (this line is the ceiling)",
                self.SERIES_COLORS[0],
                0.035,
                "bottom",
            ),
            (False, "unanswerable", self.SERIES_COLORS[1], -0.035, "top"),
        ]
        for answerable, label, color, offset, align in series:
            rows = per_question[per_question["is_answerable"] == answerable]
            # Mean within each seed first, then across seeds: the seed is the
            # sampling unit, so a seed with fewer questions must not weigh less.
            by_k = (
                rows.groupby(["top_k", "seed"], observed=True)["retrieval_hit"]
                .mean()
                .groupby("top_k", observed=True)
                .mean()
            )
            ks = [int(k) for k in by_k.index]
            rates = [float(rate) for rate in by_k]
            ax.plot(
                ks,
                rates,
                marker="o",
                linewidth=2,
                markersize=8,
                color=color,
                label=label,
            )
            for k, rate in zip(ks, rates):
                ax.text(
                    k,
                    rate + offset,
                    f"{rate:.3f}",
                    ha="center",
                    va=align,
                    fontsize=9,
                    color="#52514e",
                )

        ax.set_xlabel("top_k")
        ax.set_ylabel("retrieval hit-rate")
        ax.set_ylim(0, 1.12)
        ax.set_xticks(sorted(int(k) for k in per_question["top_k"].unique()))
        ax.set_title("Retrieval hit-rate vs top_k", fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        fig.tight_layout()
        fig.savefig(self._figure_path / "retrieval_hit_rate.png", dpi=150)
        plt.show()

    def plot_similarity_distribution(self, frame: pd.DataFrame) -> None:
        """Grid of top-1 similarity histograms: one row per model, one column
        per RAG configuration.

        Motivates (or rules out) a similarity threshold as a second tuning axis:
        if wrong answers cluster at low similarity, declining below a cut-off
        would help without ever calling the LLM.

        Split rather than pooled, for two unrelated reasons. `plain` has no
        retriever, so its top_similarity is a 0.0 filler and not a measurement.
        And top-1 similarity is identical across k by construction - only the
        answer changes - so pooling the configs would repeat every question
        three times at the same x, inflating n with correlated rows and letting
        a single question land in both the correct and the incorrect histogram.
        """
        self._figure_path.mkdir(exist_ok=True)
        rag = frame[frame["config"] != "plain"]
        if rag.empty:
            print("No RAG answers to plot.")
            return

        models = model_order(rag)
        configs = [config for config in config_order(rag) if config != "plain"]

        # Bins come from every panel's data at once. Per-panel bins would give
        # each cell its own x range, so a bar in the same place would mean a
        # different similarity in each cell - which is the one thing a grid has
        # to make safe to compare.
        low, high = rag["top_similarity"].min(), rag["top_similarity"].max()
        pad = max((high - low) * 0.05, 0.01)
        bins = np.linspace(low - pad, high + pad, 21).tolist()

        fig, axes = plt.subplots(
            len(models),
            len(configs),
            figsize=(4.2 * len(configs), 3.4 * len(models)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for row, model in enumerate(models):
            for col, config in enumerate(configs):
                ax = axes[row][col]
                cell = rag[(rag["llm_model"] == model) & (rag["config"] == config)]
                correct = cell.loc[cell["correct"], "top_similarity"].tolist()
                incorrect = cell.loc[~cell["correct"], "top_similarity"].tolist()

                ax.hist(
                    [correct, incorrect],
                    bins=bins,
                    density=True,
                    rwidth=0.85,
                    color=[self.SERIES_COLORS[0], self.SERIES_COLORS[1]],
                )

                # Density normalises each group away from its own size, so the
                # counts have to be written down somewhere: a tall bar built
                # from four answers must not read like one built from four
                # hundred.
                ax.text(
                    0.97,
                    0.94,
                    f"F1 >= {CORRECT_F1_THRESHOLD}: {len(correct)}\n"
                    f"F1 <  {CORRECT_F1_THRESHOLD}: {len(incorrect)}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#52514e",
                )

                # Label the edges only: an inner cell inherits both its
                # column's config and its row's model from the panels around it.
                if row == 0:
                    ax.set_title(config, fontsize=10)
                if row == len(models) - 1:
                    ax.set_xlabel("top-1 cosine similarity")
                if col == 0:
                    ax.set_ylabel(model, fontsize=9)

        # Proxy handles rather than the bar containers: a cell where one group
        # is empty returns an empty container, and indexing into it would raise.
        legend_handles = [
            Patch(color=self.SERIES_COLORS[0]),
            Patch(color=self.SERIES_COLORS[1]),
        ]
        fig.legend(
            legend_handles,
            [f"F1 >= {CORRECT_F1_THRESHOLD}", f"F1 < {CORRECT_F1_THRESHOLD}"],
            frameon=False,
            fontsize=9,
            loc="lower center",
            ncol=2,
        )
        fig.suptitle(
            "Retrieval similarity: correct vs incorrect answers",
            fontsize=11,
            x=0.01,
            ha="left",
        )
        fig.supylabel("density (each group normalised separately)", fontsize=9)
        # Reserve the bottom strip for the shared legend
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        fig.savefig(self._figure_path / "similarity_distribution.png", dpi=150)
        plt.show()

    def plot_similarity_ecdf(self, frame: pd.DataFrame) -> None:
        """ECDF of top-1 similarity, correct vs incorrect, on the same grid.

        The histogram shows the shape; this one answers the actual question.
        A rule of the form "decline below this similarity" is read straight
        off the y axis: at any cut-off, the pink curve is the share of wrong
        answers the rule would stop and the blue curve the share of right ones
        it would throw away. The rule is only worth having where pink runs
        clearly above blue - curves that travel together mean the threshold
        cannot separate the two groups at any value.

        An ECDF rather than a second histogram because it takes no bin width,
        so nothing in the picture is an artefact of a binning choice, and two
        monotone curves are easier to compare than two bar fields.
        Source: https://seaborn.pydata.org/tutorial/distributions.html
        """
        self._figure_path.mkdir(exist_ok=True)
        rag = frame[frame["config"] != "plain"]
        if rag.empty:
            print("No RAG answers to plot.")
            return

        models = model_order(rag)
        # `config_order` reads the column's categories, which survive a row
        # filter, so "plain" is still listed here even though no row has it.
        configs = [config for config in config_order(rag) if config != "plain"]

        fig, axes = plt.subplots(
            len(models),
            len(configs),
            figsize=(4.2 * len(configs), 3.4 * len(models)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for row, model in enumerate(models):
            for col, config in enumerate(configs):
                ax = axes[row][col]
                cell = rag[(rag["llm_model"] == model) & (rag["config"] == config)]
                correct = cell.loc[cell["correct"], "top_similarity"].tolist()
                incorrect = cell.loc[~cell["correct"], "top_similarity"].tolist()

                for values, color in (
                    (correct, self.SERIES_COLORS[0]),
                    (incorrect, self.SERIES_COLORS[1]),
                ):
                    if values:
                        ax.ecdf(values, color=color, linewidth=2)

                # The largest vertical gap between the curves is the best any
                # threshold on this axis can do. Naming it turns "the lines
                # look close" into a number the report can quote.
                gap, at = _largest_ecdf_gap(correct, incorrect)
                ax.text(
                    0.97,
                    0.06,
                    f"best gap {gap:.2f} @ {at:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#52514e",
                )

                if row == 0:
                    ax.set_title(config, fontsize=10)
                if row == len(models) - 1:
                    ax.set_xlabel("top-1 cosine similarity")
                if col == 0:
                    ax.set_ylabel(model, fontsize=9)

        legend_handles = [
            Patch(color=self.SERIES_COLORS[0]),
            Patch(color=self.SERIES_COLORS[1]),
        ]
        fig.legend(
            legend_handles,
            [
                f"F1 >= {CORRECT_F1_THRESHOLD} (would be lost)",
                f"F1 < {CORRECT_F1_THRESHOLD} (would be stopped)",
            ],
            frameon=False,
            fontsize=9,
            loc="lower center",
            ncol=2,
        )
        fig.suptitle(
            "Would a similarity threshold work? Share of each group at or below x",
            fontsize=11,
            x=0.01,
            ha="left",
        )
        fig.supylabel("cumulative share of the group", fontsize=9)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        fig.savefig(self._figure_path / "similarity_ecdf.png", dpi=150)
        plt.show()

    def plot_token_cost(self, frame: pd.DataFrame) -> None:
        """Stacked prompt + completion tokens per question, one panel per model.

        Stacked rather than two charts because the two parts move in opposite
        directions: retrieval buys a bigger prompt and pays for it with a
        shorter completion, and only the stack shows both the trade and the
        total it nets out to.

        Averaged per seed first, then across seeds, like every other figure
        here - the seed is the sampling unit, so the error bar on the total
        is the spread between question samples, not between questions.
        """
        self._figure_path.mkdir(exist_ok=True)
        configs = config_order(frame)
        models = model_order(frame)

        fig, axes = plt.subplots(
            1,
            len(models),
            figsize=(5.2 * len(models), 4.5),
            sharey=True,
            squeeze=False,
        )

        for col, model in enumerate(models):
            ax = axes[0][col]
            prompt, completion, totals, spread = [], [], [], []
            for config in configs:
                per_seed_prompt = _mean_per_seed(frame, model, config, "prompt_tokens")
                per_seed_completion = _mean_per_seed(
                    frame, model, config, "completion_tokens"
                )
                per_seed_total = _mean_per_seed(frame, model, config, "total_tokens")
                prompt.append(
                    float(np.mean(per_seed_prompt)) if per_seed_prompt else 0.0
                )
                completion.append(
                    float(np.mean(per_seed_completion)) if per_seed_completion else 0.0
                )
                totals.append(float(np.mean(per_seed_total)) if per_seed_total else 0.0)
                spread.append(float(np.std(per_seed_total)) if per_seed_total else 0.0)

            x = np.arange(len(configs))
            # edgecolor in the surface colour reads as a gap between the two
            # segments rather than as a border drawn around them.
            ax.bar(
                x,
                prompt,
                0.62,
                color=self.SERIES_COLORS[0],
                edgecolor="#fcfcfb",
                linewidth=1.5,
            )
            ax.bar(
                x,
                completion,
                0.62,
                bottom=prompt,
                color=self.SERIES_COLORS[2],
                edgecolor="#fcfcfb",
                linewidth=1.5,
                yerr=spread,
                capsize=3,
                error_kw={"ecolor": "#52514e", "elinewidth": 1},
            )

            # One label per bar - the total, which is the number the report
            # quotes. The split is readable off the segments and the axis.
            for xi, total, std in zip(x, totals, spread):
                ax.text(
                    xi,
                    total + std + max(totals) * 0.03,
                    f"{total:,.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#52514e",
                )

            ax.set_xticks(x, configs)
            ax.set_title(model, fontsize=10, loc="left")
            if col == 0:
                ax.set_ylabel("mean tokens per question")

        top = max(
            float(np.mean(_mean_per_seed(frame, model, config, "total_tokens") or [0]))
            for model in models
            for config in configs
        )
        for ax in axes[0]:
            ax.set_ylim(0, top * 1.18)

        legend_handles = [
            Patch(color=self.SERIES_COLORS[0]),
            Patch(color=self.SERIES_COLORS[2]),
        ]
        fig.legend(
            legend_handles,
            ["prompt tokens", "completion tokens"],
            frameon=False,
            fontsize=9,
            loc="lower center",
            ncol=2,
        )
        n_seeds = len(seed_order(frame))
        fig.suptitle(
            "Token cost per question by configuration\n"
            f"stacked mean, error bar is std of the total across {n_seeds} seed(s)",
            fontsize=11,
            x=0.01,
            ha="left",
        )
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        fig.savefig(self._figure_path / "token_cost.png", dpi=150)
        plt.show()


# %% [markdown]
# ### SECTION 7.3: Summary tables for analysis
#

# %%
# What each quadrant of the grounding cross-tab means, so the table can be
# read without going back to the docstring.
GROUNDING_MEANING = {
    (True, True): "grounded and right",
    (True, False): "copied the wrong passage",
    (False, True): "right but reworded (metric blind spot)",
    (False, False): "invented (hallucination)",
}


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Every metric named in the brief, one row per model and configuration.

    Returned as a DataFrame rather than printed, so Jupyter renders it as a
    real table and the numbers can go straight into the report.

    decline_P / decline_R treat "the question is unanswerable" as the positive
    class: precision is how often a decline was justified, recall is how many
    of the unanswerable questions the model actually declined.

    `hit` and `in_ctx` average nullable columns, so their denominators are
    only the rows where the metric applies - declines and plain runs are
    missing by design, not zero.
    """
    flagged = frame.assign(
        decline_tp=frame["declined"] & ~frame["is_answerable"],
        decline_fp=frame["declined"] & frame["is_answerable"],
        decline_fn=~frame["declined"] & ~frame["is_answerable"],
    )

    table = flagged.groupby(["llm_model", "config"], observed=True).agg(
        EM=("em", "mean"),
        F1=("f1", "mean"),
        hit=("retrieval_hit", "mean"),
        in_ctx=("supported_by_source", "mean"),
        _tp=("decline_tp", "sum"),
        _fp=("decline_fp", "sum"),
        _fn=("decline_fn", "sum"),
        lat_s=("latency_s", "mean"),
        tokens=("total_tokens", "mean"),
    )

    # 0/0 gives NaN, which is the honest answer when a model never declined
    # (no precision to report) or every question was answerable (no recall).
    table["decline_P"] = table["_tp"] / (table["_tp"] + table["_fp"])
    table["decline_R"] = table["_tp"] / (table["_tp"] + table["_fn"])

    columns = ["EM", "F1", "hit", "in_ctx", "decline_P", "decline_R", "lat_s", "tokens"]
    return table[columns].astype(float).round(3)


def grounding_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of "copied from the context" against "answered correctly".

    `supported_by_source` on its own says whether the answer text appears in
    the retrieved documents - that is copying, not correctness. Crossed with
    F1 it separates the two failure modes, which have different fixes:

      copied + wrong     -> the retrieved passage was misread or was a distractor
      not copied + wrong -> the model ignored the context and invented an answer

    Only RAG answers that were actually given are counted. `scored` and
    `declined` are repeated on every row on purpose: `share` is out of the
    answers a model chose to give, so the shares are NOT comparable across
    models until you can see how many each one declined. A model that answers
    twice and copies well both times shows 100% here.
    """
    attempted = frame[frame["config"] != "plain"]
    scored = attempted.dropna(subset=["supported_by_source"])
    if scored.empty:
        return pd.DataFrame()

    # Safe after dropna, and it turns the nullable boolean into a plain one
    # so the quadrant lookup below matches on real True / False.
    scored = scored.assign(copied=scored["supported_by_source"].astype(bool))

    counts = scored.groupby(["llm_model", "copied", "correct"], observed=True).size()
    # Give every model all four quadrants, including the ones with no cases:
    # a zero in "invented (hallucination)" is a result worth showing.
    quadrants = pd.MultiIndex.from_product(
        [sorted(scored["llm_model"].unique()), [True, False], [True, False]],
        names=["llm_model", "copied", "correct"],
    )
    table = counts.reindex(quadrants, fill_value=0).rename("n").reset_index()

    n_scored = scored.groupby("llm_model", observed=True).size()
    n_attempted = attempted.groupby("llm_model", observed=True).size()
    table["scored"] = table["llm_model"].map(n_scored)
    table["declined"] = table["llm_model"].map(n_attempted - n_scored)
    table["share"] = (table["n"] / table["scored"]).round(3)
    table["failure_mode"] = [
        GROUNDING_MEANING[(copied, correct)]
        for copied, correct in zip(table["copied"], table["correct"])
    ]

    return table.set_index(["llm_model", "copied", "correct"])[
        ["n", "share", "scored", "declined", "failure_mode"]
    ]


HYDRATED_COLUMNS = ("question", "gold_answers", "gold_context", "retrieved_context")


def _require_hydrated(frame: pd.DataFrame) -> None:
    """Fail loudly rather than return a table of ids nobody can read."""
    missing = [column for column in HYDRATED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(
            f"call hydrate(frame, Dataset()) first - missing columns: {missing}"
        )


def _snippet(passages: list[str], width: int = 240) -> str:
    """First retrieved passage, trimmed to something quotable."""
    if not passages:
        return ""
    text = " ".join(passages[0].split())
    return text if len(text) <= width else text[: width - 1] + "…"


def find_failure_examples(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Hallucination candidates for the report: answers invented for
    questions that have no answer (the system did not decline).

    The gold answer is empty here by construction - the question has none -
    so what makes these rows explainable is the passage the model was looking
    at while it invented something. That is why the frame has to be hydrated.
    """
    _require_hydrated(frame)
    failures = frame[~frame["is_answerable"] & ~frame["declined"]].copy()
    failures["top_context"] = failures["retrieved_context"].map(_snippet)
    # A plain run retrieved nothing, so its row can be reported but not
    # diagnosed. Sort those last so `head` does not spend every slot on rows
    # whose top_context is blank.
    failures = failures.sort_values(
        "top_context", key=lambda column: column.eq(""), kind="stable"
    )
    columns = [
        "llm_model",
        "config",
        "question",
        "answer",
        "top_similarity",
        "top_context",
    ]
    return failures[columns].head(n)


def find_wrong_answer_examples(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Answerable questions the model answered anyway, and got wrong.

    The complement of `find_failure_examples`, and the one that carries the
    gold answer: here a right answer exists, so the row can be diagnosed
    instead of merely reported. `retrieval_hit` splits the two causes that
    need different fixes - False means the retriever never surfaced the right
    passage, True means the model had it and still misread it.

    Sorted so the retrieval_hit=True rows come first: those are the ones that
    indict the model rather than the index, and they are the harder finding.
    """
    _require_hydrated(frame)
    wrong = frame[
        frame["is_answerable"] & ~frame["declined"] & ~frame["correct"]
    ].copy()
    wrong["top_context"] = wrong["retrieved_context"].map(_snippet)
    wrong = wrong.sort_values("retrieval_hit", ascending=False, na_position="last")
    columns = [
        "llm_model",
        "config",
        "question",
        "gold_answers",
        "answer",
        "f1",
        "retrieval_hit",
        "top_similarity",
        "top_context",
    ]
    return wrong[columns].head(n)


def threshold_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The best "decline below this similarity" rule available, per cell.

    One row per model and configuration, reporting the cut-off that maximises
    (wrong answers stopped) - (right answers lost), and what that cut-off
    actually buys. Reading it: `net` is the ceiling on the whole idea, not the
    score of one guess, because the cut-off was chosen to maximise it on this
    very data - a real deployment would pick the threshold on one split and
    pay for it on another, so the honest number is somewhat worse than shown.
    A net near zero means no threshold on this axis is worth having.
    """
    rag = frame[frame["config"] != "plain"]
    if rag.empty:
        return pd.DataFrame()

    rows = []
    for (model, config), cell in rag.groupby(["llm_model", "config"], observed=True):
        correct = cell.loc[cell["correct"], "top_similarity"].tolist()
        incorrect = cell.loc[~cell["correct"], "top_similarity"].tolist()
        net, cutoff = _largest_ecdf_gap(correct, incorrect)
        stopped = (
            float(np.mean(np.asarray(incorrect) <= cutoff))
            if incorrect
            else float("nan")
        )
        lost = (
            float(np.mean(np.asarray(correct) <= cutoff)) if correct else float("nan")
        )
        rows.append(
            {
                "llm_model": model,
                "config": config,
                "cutoff": cutoff,
                "stopped_wrong": stopped,
                "lost_correct": lost,
                "net": net,
                "n_wrong": len(incorrect),
                "n_correct": len(correct),
            }
        )

    table = pd.DataFrame(rows).set_index(["llm_model", "config"])
    return table.round(3)


def cost_table(frame: pd.DataFrame) -> pd.DataFrame:
    """What each configuration costs, and what that spend buys.

    `vs_plain` is the total-token multiplier against the no-RAG baseline for
    the same model, so each model is compared to its own baseline rather than
    to the other model's. `f1_per_1k` is the efficiency column: quality per
    thousand tokens, which is the only place cost and accuracy are read
    together - a configuration can win on F1 and still lose here.
    """
    grouped = frame.groupby(["llm_model", "config"], observed=True)
    table = grouped.agg(
        prompt=("prompt_tokens", "mean"),
        completion=("completion_tokens", "mean"),
        total=("total_tokens", "mean"),
        lat_s=("latency_s", "mean"),
        F1=("f1", "mean"),
    )

    baseline = table.xs("plain", level="config")["total"]
    table["vs_plain"] = [
        total / baseline[model] for (model, _), total in table["total"].items()
    ]
    # Quality per thousand tokens: the trade-off the report has to argue.
    table["f1_per_1k"] = table["F1"] / table["total"] * 1000

    columns = ["prompt", "completion", "total", "vs_plain", "lat_s", "F1", "f1_per_1k"]
    return table[columns].round(
        {
            "prompt": 0,
            "completion": 0,
            "total": 0,
            "vs_plain": 2,
            "lat_s": 2,
            "F1": 3,
            "f1_per_1k": 3,
        }
    )


# %% [markdown]
# ## SECTION 8: Run the analysis and generate plots and tables
#

# %%
results_frame = results_to_frame(results)
print(
    f"{len(results_frame)} records | models: {model_order(results_frame)} "
    f"| configs: {config_order(results_frame)} | seeds: {seed_order(results_frame)}"
)

# Join the question text, gold answers and passages back onto the records.
# Any Dataset works here: the corpus and the id lookup are seed-independent,
# so this recovers the prose offline from the stored ids - no API calls.
results_frame = hydrate(results_frame, Dataset())
print(f"hydrated columns: {[c for c in HYDRATED_COLUMNS]}")

# %% [markdown]
# ### SECTION 8.1: Summary of every metric in the brief
#

# %%
summary_table(results_frame)

# %% [markdown]
# ### SECTION 8.2: Where the grounded answers went wrong
#

# %%
grounding_breakdown(results_frame)

# %% [markdown]
# ### SECTION 8.3: Figures
#

# %%
plot = Plot(results_frame)
# F1 for all questions - Answerable and unanswerable
plot.plot_metric_by_config(results_frame, "f1")
# Exact match for all questions - Answerable and unanswerable
plot.plot_metric_by_config(results_frame, "em")
# F1 for answerable questions only
plot.plot_metric_by_config(results_frame, "f1", subset=True)
# F1 for unanswerable questions only
plot.plot_metric_by_config(results_frame, "f1", subset=False)
# Retrieval hit-rate: share of questions whose own context made it into the top-k
plot.plot_retrieval_hit_rate(results_frame)

plot.plot_similarity_distribution(results_frame)
plot.plot_similarity_ecdf(results_frame)
plot.plot_token_cost(results_frame)

# %% [markdown]
# ### SECTION 8.4: Concrete failures to quote in the report
#

# %%
find_failure_examples(results_frame)

# %% [markdown]
# ### SECTION 8.5: Answerable questions the model still got wrong
#

# %%
find_wrong_answer_examples(results_frame)

# %% [markdown]
# ### SECTION 8.6: Would a similarity threshold have helped?
#

# %%
threshold_table(results_frame)

# %% [markdown]
# ### SECTION 8.7: What each configuration costs, and what it buys
#

# %%
cost_table(results_frame)
