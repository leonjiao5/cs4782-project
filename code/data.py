def load_aime(year: int):
    raise NotImplementedError


def load_train_corpus(name: str):
    raise NotImplementedError


def format_prompt(problem: str) -> str:
    raise NotImplementedError


def extract_boxed_answer(text: str) -> str:
    raise NotImplementedError
