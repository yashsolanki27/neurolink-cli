from llm_client import LLMClient

MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",
    "inclusionai/ling-3.0-flash:free",
]


def choose_model():
    print("Available models:")
    for i, m in enumerate(MODELS, start=1):
        print(f"{i}. {m}")
    choice = input("Choose a model number (or press Enter for default): ")
    if choice.strip().isdigit():
        index = int(choice) - 1
        if 0 <= index < len(MODELS):
            return MODELS[index]
    return MODELS[0]


def run():
    model = choose_model()
    client = LLMClient(model=model)
    print(f"Using: {client}")
    print("Type 'bye' to exit.\n")

    while True:
        prompt = input("You: ")

        if prompt.strip().lower() == "bye":
            print("Goodbye!")
            break

        result = client.ask(prompt)

        if result:
            print(f"AI: {result}\n")
        else:
            print("No response received.\n")


if __name__ == "__main__":
    run()
