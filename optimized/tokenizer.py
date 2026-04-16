import tiktoken


base_encoding = tiktoken.get_encoding("r50k_base")
special_tokens = {
    "[INST]": base_encoding.n_vocab,
    "[/INST]": base_encoding.n_vocab + 1,
    "[STARTOFTEXT]": base_encoding.n_vocab + 2,
    "[ENDOFTEXT]": base_encoding.n_vocab + 3,
}

tokenizer = tiktoken.Encoding(
    name="thonk_v1_tokenizer",
    pat_str=base_encoding._pat_str,
    mergeable_ranks=base_encoding._mergeable_ranks,
    special_tokens={**base_encoding._special_tokens, **special_tokens},
)

_allowed_special = frozenset(special_tokens)


def encode(text: str) -> list[int]:
    return tokenizer.encode(
        text,
        allowed_special=_allowed_special,
        disallowed_special=(),
    )


def decode(tokens: list[int]) -> str:
    return tokenizer.decode(tokens)
