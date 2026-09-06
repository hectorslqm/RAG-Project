# %% ### SECTION 1: EMBEDDINGS AND VECTOR SEARCH
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
        similarities = np.dot(self._embeddings, query_vec)

        # Get top_k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                (self._documents[idx], float(similarities[idx]), self._metadata[idx])
            )

        return results


# %% ### SECTION 2: LLM Models with two providers + SQuAD Dataset
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

    model: str
    provider: ClassVar[str]
    _registry: ClassVar[dict[str, type["LLM"]]] = {}

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
            return LLM._registry[provider.upper()](model_name)
        except KeyError:
            raise ValueError(f"Unsupported model provider: {provider}")


class OpenAILLM(LLM):
    """
    A subclass of LLM to interact specifically with OpenAI's language models.
    **IMPORTANT**:
        Make sure to set the OPENAI_API_KEY environment variable before using this class.
    Args:
        model (str): The name of the OpenAI model to be used.

    Returns:
        OpenAILLM: An instance of the OpenAILLM class initialized with the specified model.
    """

    # Define the provider for the OpenAILLM class
    provider: ClassVar[str] = "OPENAI"

    API_KEY = os.getenv("OPENAI_API_KEY")

    def __init__(self, model: str):
        self._client = OpenAI(api_key=self.API_KEY)
        self._model = model

    def generate(self, prompt: str) -> LLMResults:
        start = time.perf_counter()
        response = self._client.responses.create(
            model=self._model,
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
        model (str): The name of the NVIDIA model to be used.

    Returns:
        NVidiaLLM: An instance of the NVidiaLLM class initialized with the specified model.
    """

    # Define the provider for the NVidiaLLM class
    provider: ClassVar[str] = "NVIDIA"

    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    API_KEY = os.getenv("NVIDIA_API_KEY")

    def __init__(self, model: str):
        self._client = OpenAI(base_url=self.NVIDIA_BASE_URL, api_key=self.API_KEY)
        self._model = model

    def generate(self, prompt: str) -> LLMResults:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model,
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
            answer=response.choices[0].message.content,
            latency_s=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


# %% ## SECTION 2.1: SQuAD Dataset
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
                id=row["id"],
                question=row[QUESTION],
                # Remove duplicated answers while preserving order
                answers=list(dict.fromkeys(row[ANSWERS][TEXT])),
                # Store the corresponding id of the context in the corpus
                context_id=self._context_to_id[row[CONTEXT]],
            )
            self._questions.append(question)
            self._questions_by_id[question.id] = question

    def get_corpus(self) -> list[str]:
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


# %% ### SECTION 3: RETRIEVAL-AUGMENTED GENERATION (RAG)
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
            answer=answer,
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


## %% ### SECTION 4: EVALUATION FRAMEWORK


class RAGEvaluator:
    """Evaluate a RAG system against test questions."""

    def __init__(self, rag_system: SimpleRAGSystem):
        self._rag_system = rag_system

    # def evaluate(self, test_cases: list[dict]) -> dict:
    #     """Evaluate RAG system on test cases.

    #     Args:
    #         test_cases: List of dictionaries with keys:
    #             - question: str
    #             - expected_answer: str (optional)
    #             - should_be_found: bool (is the answer in the document set?)
    #             - context: str (the expected source text)

    #     Returns:
    #         Evaluation metrics
    #     """
    #     results = {
    #         "total": len(test_cases),
    #         "correct": 0,
    #         "hallucinations": 0,
    #         "citations_present": 0,
    #         "details": [],
    #     }

    #     for case in test_cases:
    #         response = self._rag_system.answer(case["question"])

    #         # Check if the answer is supported by context
    #         is_supported = self._check_support(response, case.get("context", ""))

    #         # Check if the system correctly declined when answer not present
    #         if not case.get("should_be_found", True):
    #             is_correct = self._check_declined(response.answer)
    #         else:
    #             is_correct = is_supported

    #         # Check for hallucinations (unsupported claims)
    #         has_hallucination = not is_supported and case.get("should_be_found", True)

    #         # Check if citations are present
    #         has_citations = len(response.sources) > 0

    #         if is_correct:
    #             results["correct"] += 1
    #         if has_hallucination:
    #             results["hallucinations"] += 1
    #         if has_citations:
    #             results["citations_present"] += 1

    #         results["details"].append(
    #             {
    #                 "question": case["question"],
    #                 "answer": response.answer,
    #                 "is_correct": is_correct,
    #                 "has_hallucination": has_hallucination,
    #                 "has_citations": has_citations,
    #                 "sources": response.sources,
    #             }
    #         )

    #     # Compute metrics
    #     results["accuracy"] = results["correct"] / results["total"]
    #     results["citation_rate"] = results["citations_present"] / results["total"]

    #     return results

    # def _check_support(self, response: RAGResponse, expected_context: str) -> bool:
    #     """Check if the answer is supported by the retrieved context."""
    #     if not expected_context:
    #         return True  # No expected context to check

    #     # Simplified check: does the answer mention content from context?
    #     # In practice, use more sophisticated methods
    #     return any(
    #         expected_context.lower() in doc.lower() for doc in response.context_used
    #     )

    # def _check_declined(self, answer: str) -> bool:
    #     """Check if the system correctly declined to answer."""
    #     decline_phrases = [
    #         "i don't know",
    #         "cannot answer",
    #         "not provided",
    #         "not available",
    #         "i cannot",
    #     ]
    #     return any(phrase in answer.lower() for phrase in decline_phrases)
