import os
from llm_client import LLMClient
from file_utils import read_prompt, save_response


def run():
    client = LLMClient()
    print(f"Using: {client}")

    if os.path.exists("prompt.txt") and os.path.getsize("prompt.txt") > 0:
        use_file = input("prompt.txt found. Use it? (y/n): ")
        if use_file.lower() == "y":
            prompt = read_prompt("prompt.txt")
        else:
            prompt = input("Enter your prompt: ")
    else:
        prompt = input("Enter your prompt: ")

    result = client.ask(prompt)

    if result:
        save_response("response.txt", result)
        print("Response saved to response.txt")
        print(result)
    else:
        print("No response received.")


if __name__ == "__main__":
    run()
