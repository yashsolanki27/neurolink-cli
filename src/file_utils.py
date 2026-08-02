def read_prompt(filename):
    with open(filename, "r") as f:
        return f.read().strip()


def save_response(filename, content):
    with open(filename, "w") as f:
        f.write(content)
