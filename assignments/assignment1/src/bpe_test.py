import unittest
import tempfile
import os
from bpe import Bpe

class TestBpe(unittest.TestCase):
    def setUp(self):
        # Create temporary files for testing
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.test_dir)

    def _create_test_file(self, content):
        """Helper to create a test file with given content"""
        filepath = os.path.join(self.test_dir, "test.txt")
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath

    def xtest_initialization(self):
        filepath = self._create_test_file("abc")
        bpe = Bpe(filepath, 256, [])
        self.assertEqual(bpe.filepath, filepath)
        self.assertEqual(bpe.vocab_size, 256)
        self.assertEqual(bpe.special_tokens, [])

    def xtest_small(self):
        test_input = """
        low low low low low
        lower lower widest widest widest
        newest newest newest newest newest newest
        """
        filepath = self._create_test_file(test_input)
        bpe = Bpe(filepath, 263, ["<|endoftext|>"])  # 256 bytes + 1 special + 6 merges
        bpe.train()

        # Check that special token and base bytes are in vocab
        self.assertIn(b'<|endoftext|>', bpe.token_to_int)

        # Check that all single bytes 0-255 are in vocab
        for i in range(256):
            self.assertIn(bytes([i]), bpe.token_to_int)

        # Should have exactly 263 tokens
        self.assertEqual(len(bpe.token_to_int), 263)

    def xtest_vocab_initialization(self):
        filepath = self._create_test_file("abacaba")
        bpe = Bpe(filepath, 260, ["<|endoftext|>"])
        bpe.train()

        # Check that special token is first in vocab
        self.assertIn(b"<|endoftext|>", bpe.token_to_int)
        self.assertEqual(bpe.token_to_int[b"<|endoftext|>"], 0)

        # Check that all bytes 0-255 are in vocab
        for i in range(256):
            self.assertIn(bytes([i]), bpe.token_to_int)

        # Check int_to_token mapping is consistent
        for token, idx in bpe.token_to_int.items():
            self.assertEqual(bpe.int_to_token[idx], token)

    def xtest_train_simple(self):
        filepath = self._create_test_file("abacaba")
        bpe = Bpe(filepath, 259, [])  # 256 base bytes + 3 merges
        bpe.train()

        # Vocab size should be 259 (256 bytes + 3 merges)
        self.assertEqual(len(bpe.vocab_mapping()), 259)

        # Should have performed some merges
        # With "abacaba", common pairs should be merged
        # Exact merges depend on frequency counts

    def xtest_train_with_special_tokens(self):
        filepath = self._create_test_file("abacaba<|endoftext|>")
        bpe = Bpe(filepath, 260, ["<|endoftext|>"])  # 1 special + 256 bytes + 3 merges
        bpe.train()

        self.assertEqual(len(bpe.vocab_mapping()), 260)

        # Check that special token is in the vocab at index 0
        self.assertIn(b"<|endoftext|>", bpe.token_to_int)
        self.assertEqual(bpe.token_to_int[b"<|endoftext|>"], 0)

    def xtest_vocab_mapping(self):
        filepath = self._create_test_file("abc")
        bpe = Bpe(filepath, 258, [])  # 256 bytes + 2 merges
        bpe.train()
        vocab = bpe.token_to_int
        self.assertIsInstance(vocab, dict)
        self.assertEqual(len(vocab), 258)
        # Check if initial bytes are in the vocab
        self.assertIn(b'a', vocab)
        self.assertIn(b'b', vocab)
        self.assertIn(b'c', vocab)

    def xtest_empty_input(self):
        filepath = self._create_test_file("")
        bpe = Bpe(filepath, 256, [])
        bpe.train()
        # Should have 256 base bytes even with empty input
        self.assertEqual(len(bpe.vocab_mapping()), 256)

    def xtest_single_character_input(self):
        filepath = self._create_test_file("aaaaa")
        bpe = Bpe(filepath, 258, [])  # 256 bytes + 2 merges
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 258)
        self.assertIn(b'a', bpe.token_to_int)
        # Should have merged 'aa' since it appears frequently
        self.assertIn(b'aa', bpe.token_to_int)

    def xtest_no_repeated_pairs(self):
        filepath = self._create_test_file("abcdefg")
        bpe = Bpe(filepath, 256, [])  # Just base bytes, no merges
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 256)
        # No merges should happen as vocab size equals base bytes
        self.assertNotIn(b'ab', bpe.token_to_int)
        self.assertNotIn(b'fg', bpe.token_to_int)

    def xtest_vocab_size_equal_to_initial_tokens(self):
        filepath = self._create_test_file("abc")
        bpe = Bpe(filepath, 256, [])  # Just base bytes
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 256)
        self.assertIn(b'a', bpe.token_to_int)
        self.assertIn(b'b', bpe.token_to_int)
        self.assertIn(b'c', bpe.token_to_int)
        # No merges with vocab_size = 256
        self.assertNotIn(b'ab', bpe.token_to_int)

    def xtest_vocab_size_with_merges(self):
        filepath = self._create_test_file("abcde")
        bpe = Bpe(filepath, 260, [])  # 256 bytes + 4 merges
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 260)

    def xtest_multiple_special_tokens(self):
        filepath = self._create_test_file("a<|endoftext|>b<|padding|>")
        bpe = Bpe(filepath, 260, ["<|endoftext|>", "<|padding|>"])  # 2 special + 256 bytes + 2 merges
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 260)
        self.assertIn(b'<|endoftext|>', bpe.token_to_int)
        self.assertIn(b'<|padding|>', bpe.token_to_int)
        # Special tokens should be at indices 0 and 1
        self.assertEqual(bpe.token_to_int[b'<|endoftext|>'], 0)
        self.assertEqual(bpe.token_to_int[b'<|padding|>'], 1)

    def xtest_special_token_as_substring(self):
        filepath = self._create_test_file("a<|endoftext|>b")
        bpe = Bpe(filepath, 258, ["<|endoftext|>"])  # 1 special + 256 bytes + 1 merge
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 258)
        self.assertIn(b'<|endoftext|>', bpe.token_to_int)
        self.assertEqual(bpe.token_to_int[b'<|endoftext|>'], 0)

    def test_simple_case(self):
        test_content = """low low low low low<|endoftext|>lower lower widest widest widest<|endoftext|>newest newest newest newest newest newest"""
        filepath = self._create_test_file(test_content)
        bpe = Bpe(filepath=filepath,
                  vocab_size=263,
                  special_tokens=["<|endoftext|>"],
                  debug_mode=True)  # 256 bytes + 14 merges
        bpe.train()

        # Check that all single bytes are in vocab
        #for i in range(256):
        #    self.assertIn(bytes([i]), bpe.token_to_int)

        # With frequent patterns like "low", "est", "new", etc.,
        # we should see some common bigrams merged
        # The exact merges depend on the frequency counting algorithm

if __name__ == '__main__':
    unittest.main()
