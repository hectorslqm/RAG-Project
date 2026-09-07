import os
import time
from typing import ClassVar

from attr import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class LLMResults:
    text: str
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

    _model: str
    _provider: ClassVar[str]
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
        if provider := getattr(cls, "_provider", None):
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
            return factory(model_name) # type: ignore[call-arg]
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
    _provider: ClassVar[str] = "OPENAI"

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
            text=response.output_text,
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
    _provider: ClassVar[str] = "NVIDIA"

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
            text=response.choices[0].message.content or "",
            latency_s=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


def main():
    NVIDIA_MODEL = "meta/muse-glimmer-30b"
    OPENAI_MODEL = "gpt-5.4-mini"
    nvidia = LLM.create("NVIDIA", NVIDIA_MODEL)
    openai = LLM.create("OPENAI", OPENAI_MODEL)
    models = [nvidia, openai]

    prompts = ["Who are you?", "How can you describe a RAG system?"]
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        for model in models:
            print(f"{model._model}: {model.generate(prompt)}")


if __name__ == "__main__":
    main()
