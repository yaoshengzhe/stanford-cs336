from __future__ import annotations

import argparse
import asyncio
import json
import os
import regex as re
import multiprocessing as mp
from functools import partial

from collections.abc import Iterable, Iterator
from typing import BinaryIO
from typing import Generator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(args):
    """Worker function to process a single chunk of text."""
    chunk_id, filepath, start, end, pattern, split_pattern, special_tokens = args
    tokens = []

    with open(filepath, 'r', encoding='utf-8') as f:
        f.seek(start)
        text = f.read(end - start)

        if split_pattern:
            text_chunks = re.split(split_pattern, text)
        else:
            text_chunks = [text]

        for chunk in text_chunks:
            for m in re.finditer(pattern, chunk):
                tokens.append(m.group(0))

    return chunk_id, tokens

class ChunkedTokenReader:
    def __init__(self, num_threads: int=1, show_progress: bool=False):
        self.pattern = PAT
        self.num_threads = max(1, num_threads)
        self.show_progress = show_progress

    def read_tokens(self, filepath: str, special_tokens: list[str]) -> Generator[str, None, None]:
        split_pattern = ''
        first_special_token = b''

        if special_tokens:
            first_special_token = special_tokens[0].encode('utf-8')
            escaped = [re.escape(st) for st in special_tokens]
            escaped.sort(key=lambda x: -len(x))
            split_pattern = f"({'|'.join(escaped)})"

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            boundaries = self._find_chunk_boundaries(f, self.num_threads, first_special_token)

            # Create tasks for parallel processing with chunk IDs
            tasks = []
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                tasks.append((i, filepath, start, end, self.pattern, split_pattern, special_tokens))

            total_chunks = len(tasks)
            if self.show_progress:
                print(f"Processing {total_chunks} chunk(s) with {self.num_threads} thread(s)...")

            # Use multiprocessing to process chunks in parallel
            if self.num_threads > 1 and len(tasks) > 1:
                with mp.Pool(processes=min(self.num_threads, len(tasks))) as pool:
                    # Use imap_unordered for better progress tracking
                    completed = 0
                    for chunk_id, token_list in pool.imap_unordered(process_chunk, tasks):
                        completed += 1
                        if self.show_progress:
                            print(f"Pre-tokenization progress: {completed}/{total_chunks} chunks completed ({100*completed/total_chunks:.1f}%)", end='\r')
                        for token in token_list:
                            yield token
                    if self.show_progress:
                        print(f"Pre-tokenization progress: {total_chunks}/{total_chunks} chunks completed (100.0%)")
            else:
                # Fall back to serial processing for single thread
                for i, task in enumerate(tasks):
                    chunk_id, tokens = process_chunk(task)
                    if self.show_progress:
                        print(f"Pre-tokenization progress: {i+1}/{total_chunks} chunks completed ({100*(i+1)/total_chunks:.1f}%)", end='\r')
                    for token in tokens:
                        yield token
                if self.show_progress and total_chunks > 0:
                    print(f"Pre-tokenization progress: {total_chunks}/{total_chunks} chunks completed (100.0%)")


    def _find_chunk_boundaries(
            self,
            file: BinaryIO,
            desired_num_chunks: int,
            split_special_token: bytes) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == "":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token.decode('utf-8', errors='replace'))
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


class Bpe:
    def __init__(self,
                 filepath: str = None,
                 vocab_size: int = None,
                 vocab: dict[int, bytes] = None,
                 merges: list[tuple[bytes, bytes]] = None,
                 special_tokens: list[str] = None,
                 debug_mode=False,
                 show_progress=False):
        self.filepath = filepath
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []
        self.merges = merges
        # dict[int, bytes]
        self.int_to_token = vocab
        self.debug_mode = debug_mode
        self.show_progress = show_progress

        if vocab:
            self.vocab_size = len(vocab.keys())
            # dict[bytes, int]
            self.token_to_int = {tok:id for id, tok in vocab.items()}

        if self.special_tokens:
            self.special_tokens.sort(key=lambda x: -len(x))

    def encode(self, text: str):
        # self.merges: list[tuple[bytes, bytes]
        st_set = set(self.special_tokens)
        all_tokens = []

        for chunk in self._split_tokens_preserve_special_token(text):
            if chunk in st_set:
                all_tokens.append(chunk.encode('utf-8'))
            else:
                # Apply pre-tokenization pattern first
                for m in re.finditer(PAT, chunk):
                    pre_token = m.group(0)
                    # Regular text - convert to bytes and apply merges
                    tokens = [b.to_bytes() for b in pre_token.encode('utf-8')]
                    for a, b in self.merges:
                        tokens = self._merge_key(tokens, a+b)
                    all_tokens.extend(tokens)

        return [self.token_to_int[tok] for tok in all_tokens if tok in self.token_to_int]

    def _split_tokens_preserve_special_token(self, text: str) -> Generator[str, None, None]:
        split_pattern = ''

        if self.special_tokens:
            escaped = [re.escape(st) for st in self.special_tokens]
            escaped.sort(key=lambda x: -len(x))
            split_pattern = f"({'|'.join(escaped)})"

        chunks = [text]

        if split_pattern:
            chunks = re.split(split_pattern, text)

        for chunk in chunks:
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            yield chunk

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for idx in self.encode(text):
                yield idx

    def decode(self, ids: list[int]) -> str:
        return b''.join([self.int_to_token.get(id, b'') for id in ids]).decode('utf-8', errors='replace')

    def train(self):
        self.merges = []
        self.token_to_int = {}
        self.int_to_token = {}
        self.idx = 0

        # Step 1: vocab initialization
        # ----------------------------
        # inserts special tokens
        for token in self.special_tokens:
            self._insert_token(token.encode('utf-8'))

        # inserts bytes: 0-255
        for token in range(256):
            self._insert_token(bytes([token]))

        # Step 2: pre-tokenization
        # ----------------------------
        # token_counts  -> dict[tuple[bytes], int]
        token_counts = self._pretokenization(self.filepath, self.special_tokens)

        # Step 3: merge
        # ----------------------------
        num_merges = self.vocab_size - self.idx
        for merge_round in range(num_merges):
            if self.show_progress:
                print(f'Merge round {merge_round + 1}/{num_merges}', end='\r')
            if self.debug_mode:
                print(f'#{merge_round} merge: {token_counts}')

            token_counts = self._merge(token_counts)

        if self.show_progress:
            print(f'Completed {num_merges} merge rounds')

    def _insert_token(self, token: bytes):
        self.token_to_int[token] = self.idx
        self.int_to_token[self.idx] = token
        self.idx += 1

    def _merge(self, token_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
        freq = {} # dict[tuple[bytes], int]
        # count byte pairs
        for token_bytes, token_count in token_counts.items():
            for byte_pair, pair_count in self._count_byte_pair(token_bytes).items():
                freq[byte_pair] = freq.get(byte_pair, 0) + pair_count * token_count

        if self.debug_mode:
            print(f'freq: {freq}')

        most_freq_token_pair, count = max(freq.items(), key=lambda x: (x[1], x[0]))

        if self.debug_mode:
            print(f'most frequent byte pair: {most_freq_token_pair}, count: {count}')

        merged_pair = b''.join(most_freq_token_pair)

        self._insert_token(merged_pair)

        new_token_counts = {}
        for token_bytes, token_count in token_counts.items():
            new_token_bytes = self._merge_key(token_bytes, merged_pair)
            new_token_counts[new_token_bytes] = new_token_counts.get(new_token_bytes, 0) + token_count

        self.merges.append((most_freq_token_pair[0], most_freq_token_pair[1]))

        return new_token_counts

    def vocab_mapping(self) -> dict[int, bytes]:
        return self.int_to_token

    def _merge_key(self, token_bytes: tuple[bytes], merged_pair: bytes) -> tuple[bytes]:
        i = 0
        new_token_bytes = []
        while i < len(token_bytes) - 1:
            if b''.join([token_bytes[i], token_bytes[i+1]]) == merged_pair:
                new_token_bytes.append(merged_pair)
                i += 2
            else:
                new_token_bytes.append(token_bytes[i])
                i += 1

        while i < len(token_bytes):
            new_token_bytes.append(token_bytes[i])
            i += 1

        return tuple(new_token_bytes)

    def _pretokenization(self, filepath: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
        # dict[tuple[bytes], int]
        token_counts = {}

        # Use multiple threads if available (defaults to CPU count)
        num_threads = mp.cpu_count() if not hasattr(self, 'num_threads') else getattr(self, 'num_threads', mp.cpu_count())

        print(f'Pre-tokenization using {num_threads} CPU(s)')

        # Pass show_progress flag to ChunkedTokenReader
        reader = ChunkedTokenReader(num_threads=num_threads, show_progress=self.show_progress)

        token_count = 0
        for token in reader.read_tokens(filepath, special_tokens):
            #token = token.strip()
            byte_token = tuple([tok.encode('utf-8') for tok in token])
            token_counts[byte_token] = token_counts.get(byte_token, 0) + 1
            token_count += 1
            if self.show_progress and token_count % 10000 == 0:
                print(f"Processed {token_count:,} tokens...", end='\r')

        if self.show_progress:
            print(f"Pre-tokenization complete: {token_count:,} total tokens, {len(token_counts):,} unique tokens")

        return token_counts

    def _count_byte_pair(self, token_bytes: tuple[bytes]) -> dict[tuple[bytes], int]:
        freq = {}
        for a, b in zip(token_bytes, token_bytes[1:]):
            k = (a, b)
            freq[k] = freq.get(k, 0) + 1
        return freq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--dataset', type=str, help='Path to the dataset file')
    parser.add_argument('--vocab-size', type=int, default=1000, help='Target vocabulary size (default: 1000)')
    parser.add_argument('--special-tokens', nargs='*', default=['<|endoftext|>'], help='List of special tokens')
    parser.add_argument('--output-vocab', type=str, default='vocab.json', help='Output path for vocabulary JSON file (default: vocab.json)')
    parser.add_argument('--output-merges', type=str, default='merges.txt', help='Output path for merges TXT file (default: merges.txt)')
    parser.add_argument('--show-progress', action='store_true', help='Display merge round progress during training')
    parser.add_argument('--num-threads', type=int, default=mp.cpu_count(), help=f'Number of threads for parallel processing (default: {mp.cpu_count()})')

    args = parser.parse_args()

    # Create and train BPE model
    bpe = Bpe(filepath=args.dataset, vocab_size=args.vocab_size, special_tokens=args.special_tokens, show_progress=args.show_progress)
    bpe.num_threads = args.num_threads  # Pass num_threads to the BPE instance
    bpe.train()

    # Save vocab (int to token mapping) as JSON
    vocab_dict = {k: v.decode('utf-8', errors='replace') for k, v in bpe.int_to_token.items()}
    with open(args.output_vocab, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    # Save merges to TXT file (one merge per line, byte pair separated by single space)
    with open(args.output_merges, 'w', encoding='utf-8') as f:
        for pair_a, pair_b in bpe.merges:
            f.write(f"{pair_a.decode('utf-8', errors='replace')} {pair_b.decode('utf-8', errors='replace')}\n")

    print(f"Training complete!")
    print(f"Vocabulary saved to: {args.output_vocab}")
    print(f"Merges saved to: {args.output_merges}")
