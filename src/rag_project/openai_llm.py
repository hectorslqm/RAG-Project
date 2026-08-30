import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


def main():
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-5.4-mini",
        input="Which number is larger, 9.11 or 9.8?",
        reasoning={"effort": "medium"},
    )
    print(response.output_text)


if __name__ == "__main__":
    print("Initializing LLM client")
    main()
