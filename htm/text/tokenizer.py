from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self: 'Tokenizer',
                 path: str,
                 max_tokens: int = 1024):
        super().__init__()
        # noinspection PyNoneFunctionAssignment
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.max_tokens = max_tokens

    def encode(self, text: str):
        return self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     truncation=False,
                                     verbose=False)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
