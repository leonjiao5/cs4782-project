def load_aime(year: int):
    raise NotImplementedError


def load_train_corpus(name: str):
    raise NotImplementedError


def format_prompt(problem: str) -> str:
    raise NotImplementedError


def extract_boxed_answer(text: str) -> str | None:
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None
