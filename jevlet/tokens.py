"""A fixed byte vocabulary plus reserved structural tokens.

The scratch model deliberately avoids a learned or externally downloaded tokenizer. UTF-8 bytes
give deterministic coverage for arbitrary option strings, while the reserved IDs make the input
topology explicit.
"""

from __future__ import annotations


class ByteTokenizer:
    SPECIAL_TOKENS = (
        "<PAD>",
        "<STATE>",
        "<QUESTION>",
        "<OPTION>",
        "<END_OPTION>",
        "<DECIDE>",
    )
    PAD = 0
    STATE = 1
    QUESTION = 2
    OPTION = 3
    END_OPTION = 4
    DECIDE = 5
    BYTE_OFFSET = len(SPECIAL_TOKENS)
    VOCAB_SIZE = BYTE_OFFSET + 256

    def encode(self, text: str, max_bytes: int | None = None) -> list[int]:
        raw = text.encode("utf-8")
        if max_bytes is not None:
            raw = raw[:max_bytes]
            while raw:
                try:
                    raw.decode("utf-8")
                    break
                except UnicodeDecodeError:
                    raw = raw[:-1]
        return [self.BYTE_OFFSET + byte for byte in raw]

    def decode(self, token_ids: list[int]) -> str:
        raw = bytes(token - self.BYTE_OFFSET for token in token_ids if token >= self.BYTE_OFFSET)
        return raw.decode("utf-8", errors="replace")

    def token_name(self, token_id: int) -> str:
        if 0 <= token_id < self.BYTE_OFFSET:
            return self.SPECIAL_TOKENS[token_id]
        return bytes([token_id - self.BYTE_OFFSET]).decode("utf-8", errors="replace")
