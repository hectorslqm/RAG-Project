import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

nvidia_api_key = os.getenv("NVIDIA_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


class LLM:
    model: str

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    @staticmethod
    def create(provider: str, model_name: str) -> "LLM":
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported model provider: {provider}")
        return PROVIDERS[provider](model_name)


class OpenAILLM(LLM):
    def __init__(self, model: str):
        self._client = OpenAI(api_key=openai_api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.responses.create(
            model=self._model,
            input=prompt,
            reasoning={"effort": "medium"},
        )
        return response.output_text


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaLLM(LLM):
    def __init__(self, model: str):
        self._client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=nvidia_api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            top_p=0.95,
            max_tokens=8192,
            stream=False,
        )
        return response.choices[0].message.content


PROVIDERS = {
    "NVIDIA": NvidiaLLM,
    "OPENAI": OpenAILLM,
}


def main():
    NVIDIA_MODEL = "meta/muse-glimmer-30b"
    OPENAI_MODEL = "gpt-5.4-mini"
    nvidia = LLM.create("NVIDIA", NVIDIA_MODEL)
    openai = LLM.create("OPENAI", OPENAI_MODEL)

    prompts = ["Who are you?", "How can you describe a RAG system?"]
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print(f"{NVIDIA_MODEL}: {nvidia.generate(prompt)}")
        print(f"{OPENAI_MODEL}: {openai.generate(prompt)}")


if __name__ == "__main__":
    main()
