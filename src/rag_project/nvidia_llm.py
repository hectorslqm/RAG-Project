import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")


def main():
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    completion = client.chat.completions.create(
        model="meta/muse-glimmer-30b",
        messages=[{"role": "user", "content": "Which number is larger, 9.11 or 9.8?"}],
        temperature=1,
        top_p=0.95,
        max_tokens=8192,
        stream=False,
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    print("Initializing LLM client")
    main()
