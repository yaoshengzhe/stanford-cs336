#!/usr/bin/env python3
import time
import tempfile
import multiprocessing as mp
from bpe import ChunkedTokenReader, Bpe

def test_multiprocessing():
    # Create a test file with some sample text
    test_text = "Hello world! " * 1000 + "<|endoftext|>" + " This is a test. " * 1000

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(test_text)
        test_file = f.name

    special_tokens = ['<|endoftext|>']

    print("Testing ChunkedTokenReader with different thread counts...")

    # Test with single thread
    print("\n1. Single thread:")
    start = time.time()
    reader_single = ChunkedTokenReader(num_threads=1)
    tokens_single = list(reader_single.read_tokens(test_file, special_tokens))
    time_single = time.time() - start
    print(f"   - Time: {time_single:.4f}s")
    print(f"   - Tokens found: {len(tokens_single)}")

    # Test with multiple threads
    num_cores = mp.cpu_count()
    print(f"\n2. Multi-thread ({num_cores} threads):")
    start = time.time()
    reader_multi = ChunkedTokenReader(num_threads=num_cores)
    tokens_multi = list(reader_multi.read_tokens(test_file, special_tokens))
    time_multi = time.time() - start
    print(f"   - Time: {time_multi:.4f}s")
    print(f"   - Tokens found: {len(tokens_multi)}")

    # Verify results are the same
    print("\n3. Verification:")
    if len(tokens_single) == len(tokens_multi):
        print(f"   ✓ Token counts match: {len(tokens_single)}")
    else:
        print(f"   ✗ Token counts differ: single={len(tokens_single)}, multi={len(tokens_multi)}")

    # Calculate speedup
    if time_single > 0:
        speedup = time_single / time_multi
        print(f"\n4. Performance:")
        print(f"   - Speedup: {speedup:.2f}x")
        print(f"   - Efficiency: {(speedup/num_cores)*100:.1f}%")

    # Clean up
    import os
    os.unlink(test_file)

    print("\nTest completed!")

if __name__ == "__main__":
    test_multiprocessing()