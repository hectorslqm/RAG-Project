# %% [markdown]
# ## SECTION 1: EMBEDDINGS AND VECTOR SEARCH

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

# %%
import re
import string
from collections import Counter


def _normalize_texts(text: str) -> str:
    """Lowercase, strip punctuation and articles, and normalize whitespace.
    E.G. "The CAR runs fast." -> "car runs fast"

    Args:
        text (str): _description_

    Returns:
        str: _description_
        TODO: Investigate source to confirm if it is the standard squad normalization
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
    """TODO:"""
    if raw is None:
        return ""
    return raw.strip()


class RAGEvaluator:
    """Evaluate a RAG system against test questions.

    Adapted from the module notebook's RAGEvaluator: the substring-support
    check was replaced with the official SQuAD v2 protocol (EM / token-F1,
    max over reference answers, decline handling for unanswerable questions).

    #TODO: See: https://github.com/huggingface/evaluate/blob/main/metrics/squad_v2/squad_v2.py
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
                            f"f1={metrics['f1']:.2f} hit={retrieval_hit} latency={latency_s:>5.1f}s | {answer[:20]}..."
                        )
                    print(f"{config}: Processed")


# %% [markdown]
# ## SECTION 6: Run the experiment

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

# %% [markdown]
# ### SECTION 7.1: Build the analysis DataFrame

# %%
import matplotlib.pyplot as plt
import pandas as pd

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


# %% [markdown]
# ### SECTION 7.2: Plotting functions for analysis

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
        """Line: share of questions whose own context made it into the top-k.

        Retrieval runs before the LLM and uses the same embeddings for both
        models, so this curve is a property of the retriever alone. Rows are
        de-duplicated by (seed, top_k, question) so a question is not counted
        once per model.
        """
        self._figure_path.mkdir(exist_ok=True)
        rag = frame[frame["config"] != "plain"].dropna(subset=["retrieval_hit"])
        if rag.empty:
            print("No retrieval records to plot.")
            return

        per_question = rag.drop_duplicates(subset=["seed", "top_k", "question_id"])
        # Mean within each seed first, then across seeds: the seed is the
        # sampling unit, so a seed with fewer questions must not weigh less.
        by_k = (
            per_question.groupby(["top_k", "seed"], observed=True)["retrieval_hit"]
            .mean()
            .groupby("top_k", observed=True)
            .mean()
        )
        ks = [int(k) for k in by_k.index]
        rates = [float(rate) for rate in by_k]

        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(
            ks,
            rates,
            marker="o",
            linewidth=2,
            markersize=8,
            color=self.SERIES_COLORS[0],
        )
        for k, rate in zip(ks, rates):
            ax.text(
                k, rate + 0.03, f"{rate:.3f}", ha="center", fontsize=9, color="#52514e"
            )
        ax.set_xlabel("top_k")
        ax.set_ylabel("retrieval hit-rate")
        ax.set_ylim(0, 1.12)
        ax.set_xticks(ks)
        # A single series needs no legend box - the title names it
        ax.set_title("Retrieval hit-rate vs top_k", fontsize=11, loc="left")
        fig.tight_layout()
        fig.savefig(self._figure_path / "retrieval_hit_rate.png", dpi=150)
        plt.show()

    def plot_similarity_distribution(self, frame: pd.DataFrame) -> None:
        """Histogram of top-1 similarity for correct vs incorrect RAG answers.

        Motivates (or rules out) a similarity threshold as a second tuning axis:
        if wrong answers cluster at low similarity, declining below a cut-off
        would help without ever calling the LLM.
        """
        self._figure_path.mkdir(exist_ok=True)
        rag = frame[frame["config"] != "plain"]
        correct = rag.loc[rag["correct"], "top_similarity"].tolist()
        incorrect = rag.loc[~rag["correct"], "top_similarity"].tolist()

        values = correct + incorrect
        if not values:
            print("No RAG answers to plot.")
            return

        # Bin over the range the data actually occupies. Cosine similarity is
        # bounded 0-1, but retrieved passages cluster in a narrow band, so a
        # fixed 0-1 axis spends most of its width on empty space.
        low, high = min(values), max(values)
        pad = max((high - low) * 0.05, 0.01)
        bins = np.linspace(low - pad, high + pad, 21).tolist()

        fig, ax = plt.subplots(figsize=(7, 4))
        # Draw the two groups side by side inside each bin rather than as two
        # translucent layers: overlapping fills mix into a third colour that
        # reads as a category of its own but means nothing. rwidth leaves a
        # gap so the paired bars never touch.
        ax.hist(
            [correct, incorrect],
            bins=bins,
            density=True,
            rwidth=0.85,
            label=[f"F1 >= {CORRECT_F1_THRESHOLD}", f"F1 < {CORRECT_F1_THRESHOLD}"],
            color=[self.SERIES_COLORS[0], self.SERIES_COLORS[1]],
        )
        ax.set_xlabel("top-1 cosine similarity")
        # Each group is normalised on its own, so the two shapes stay
        # comparable even though far more answers are correct than incorrect.
        ax.set_ylabel("density (each group normalised separately)")
        ax.set_title(
            "Retrieval similarity: correct vs incorrect answers",
            fontsize=11,
            loc="left",
        )
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(self._figure_path / "similarity_distribution.png", dpi=150)
        plt.show()


# %% [markdown]
# ### SECTION 7.3: Summary tables for analysis

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


def find_failure_examples(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Hallucination candidates for the report: answers invented for
    questions that have no answer (the system did not decline)."""
    failures = frame[~frame["is_answerable"] & ~frame["declined"]]
    columns = ["llm_model", "config", "question_id", "top_similarity", "answer"]
    return failures[columns].head(n)


# %% [markdown]
# ## SECTION 8: Run the analysis and generate plots

# %%
results_frame = results_to_frame(results)
print(
    f"{len(results_frame)} records | models: {model_order(results_frame)} "
    f"| configs: {config_order(results_frame)} | seeds: {seed_order(results_frame)}"
)

# %% [markdown]
# ### SECTION 8.1: Summary of every metric in the brief

# %%
summary_table(results_frame)

# %% [markdown]
# ### SECTION 8.2: Where the grounded answers went wrong

# %%
grounding_breakdown(results_frame)

# %% [markdown]
# ### SECTION 8.3: Figures

# %%
plot = Plot(results_frame)
plot.plot_metric_by_config(results_frame, "f1")
plot.plot_metric_by_config(results_frame, "em")
plot.plot_metric_by_config(results_frame, "f1", subset=True)
plot.plot_metric_by_config(results_frame, "f1", subset=False)
plot.plot_retrieval_hit_rate(results_frame)
plot.plot_similarity_distribution(results_frame)

# %% [markdown]
# ### SECTION 8.4: Concrete failures to quote in the report

# %%
find_failure_examples(results_frame)
