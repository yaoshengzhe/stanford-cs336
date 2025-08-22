from __future__ import annotations

import re

class Bpe:
    def __init__(self, tokens: str, vocab_size: int, special_tokens: list[str]):
        self.raw_tokens = bytes(tokens, encoding='utf-8')
        self.vocab_size = vocab_size
        self.special_tokens = [ i for i in map(lambda t: bytes(t, encoding='utf-8'),
                                  special_tokens)]
        self.merges = []
        self.token_to_int = {}
        self.int_to_token = {}
        
    def train(self):
        self.merges = []
        self.token_to_int = {}
        self.int_to_token = {}

        tokens, idx = self._preprocess()

        nonspecial_vocab_size = self.vocab_size - len(self.special_tokens)

        for i in range(nonspecial_vocab_size, self.vocab_size):
            t = self.special_tokens[i-nonspecial_vocab_size]
            self.token_to_int[t] = i
            self.int_to_token[i] = t

        num_merges = self.vocab_size - len(self.special_tokens) - idx

        for merge_round in range(num_merges):
            freq = {}

            for a, b in zip(tokens, tokens[1:]):
                k = (a, b)
                freq[k] = freq.get(k, 0) + 1

            if freq:
                tokens = self._merge(tokens, freq, idx)
                idx += 1

    def _merge(self, tokens, freq: dict[(int, int), int], idx: int) -> list[int]:
        most_freq_token_pair = max(freq, key=freq.get)
        new_token = self.int_to_token[most_freq_token_pair[0]] + \
            self.int_to_token[most_freq_token_pair[1]]
        self.int_to_token[idx] = new_token
        self.token_to_int[new_token] = idx

        new_tokens = []

        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == most_freq_token_pair[0] and tokens[i+1] == most_freq_token_pair[1]:
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        if i < len(tokens):
            new_tokens.append(tokens[i])
        return new_tokens
            
    def _preprocess(self) -> tuple[list[int], int]:
        tokens = []
        idx = 0
        stream = []
        if self.special_tokens:
            regex_pattern = b'|'.join(map(re.escape, self.special_tokens))

            stream = [b for st in self.special_tokens for parts in re.split(regex_pattern, self.raw_tokens) for b in parts]
        else:
            stream = self.raw_tokens
        
        for b in stream:
            b = b.to_bytes()
            if b not in self.token_to_int:
                self.token_to_int[b] = idx
                self.int_to_token[idx] = b
                idx += 1
            tokens.append(self.token_to_int[b])
                                
        return (tokens, idx)

    def vocab_mapping(self) -> dict[int, bytes]:
        return self.int_to_token
