import random
from dataclasses import dataclass

from attrs import field
from datasets import load_dataset


@dataclass
class Dataset_question:
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

        self._questions: list[Dataset_question] = []
        self._questions_by_id: dict[str, Dataset_question] = {}
        # Create a QAPair for each row in the dataset and store it
        for row in self._dataset:
            question = Dataset_question(
                id=row["id"], # type: ignore
                question=row[QUESTION], # type: ignore
                # Remove duplicated answers while preserving order
                answers=list(dict.fromkeys(row[ANSWERS][TEXT])), # type: ignore
                # Store the corresponding id of the context in the corpus 
                context_id=self._context_to_id[row[CONTEXT]], # type: ignore
            )
            self._questions.append(question)
            self._questions_by_id[question.id] = question

    def get_corpus(self) -> list[str]:
        """Return the list of unique contexts in the corpus."""
        return self._corpus

    def get_questions(
        self, n: int | None = None, only_answerable: bool | None = None
    ) -> list[Dataset_question]:
        """
        Return a list of questions based on the specified criteria.

        n: Number of questions to return. If None (or larger than the pool), return all matching questions in dataset order.
        only_answerable: If True, return only answerable questions. If False, only unanswerable ones. If None, no filtering.

        Returns:
            A list of Dataset_question objects. When n is smaller than the pool, a random
            sample drawn from the generator, so results are reproducible.
        """
        questions = self._questions
        if only_answerable is not None:
            questions = [q for q in questions if q.is_answerable == only_answerable]
        if n is None or n >= len(questions):
            return list(questions)
        return self._random.sample(questions, n)

    def _get_question_by_id(self, question_id: str) -> Dataset_question:
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


ds = Dataset()
print(ds.statistics())
print("=" * 60)
print("sample of questions")
print("=" * 60)
sample_questions = ds.get_questions(n=5)
for i, question in enumerate(sample_questions):
    print("-" * 60)
    print(f"Question {i + 1}:")
    print("-" * 60)
    print(question)
    print("-" * 60)
    print("context")
    print("-" * 60)
    print(ds.context_for_question(question.id))
