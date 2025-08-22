import unittest
from bpe import Bpe

class TestBpe(unittest.TestCase):
    def test_initialization(self):
        bpe = Bpe("abc", 256, [])
        self.assertEqual(bpe.raw_tokens, b"abc")
        self.assertEqual(bpe.vocab_size, 256)
        self.assertEqual(bpe.special_tokens, [])

    def test_preprocess(self):
        bpe = Bpe("abacaba", 260, ["<|endoftext|>"])
        tokens, idx = bpe._preprocess()
        # The special token should not be in the preprocessed output of raw_tokens
        self.assertNotIn(bpe.token_to_int.get(b"<|endoftext|>"), tokens)
        
        # check that all unique bytes are in the vocab
        unique_bytes = sorted(list(set(b"abacaba")))
        for b in unique_bytes:
            self.assertIn(b.to_bytes(), bpe.token_to_int)
            self.assertIn(bpe.token_to_int[b.to_bytes()], bpe.int_to_token)

    def test_train_simple(self):
        bpe = Bpe("abacaba", 5, [])
        bpe.train()
        
        # Vocab size should be 5
        self.assertEqual(len(bpe.vocab_mapping()), 5)
        
        # 'a' and 'b' should be merged first, then 'ab' and 'a'
        # Check if 'ab' is in the vocab
        self.assertIn(b'ab', bpe.token_to_int)
        
        # Check if 'aba' is in the vocab
        self.assertIn(b'aba', bpe.token_to_int)

    def test_train_with_special_tokens(self):
        bpe = Bpe("abacaba<|endoftext|>", 8, ["<|endoftext|>"])
        bpe.train()
        
        self.assertEqual(len(bpe.vocab_mapping()), 8)
        
        # Check that special token is in the vocab
        self.assertIn(b"<|endoftext|>", bpe.token_to_int)
        
        # Check that the special token has the correct integer value
        self.assertEqual(bpe.token_to_int[b"<|endoftext|>"], 7)

    def test_vocab_mapping(self):
        bpe = Bpe("abc", 4, [])
        bpe.train()
        vocab = bpe.token_to_int
        self.assertIsInstance(vocab, dict)
        self.assertEqual(len(vocab), 4)
        # Check if initial bytes are in the vocab
        self.assertIn(b'a', vocab)
        self.assertIn(b'b', vocab)
        self.assertIn(b'c', vocab)

    def test_empty_input(self):
        bpe = Bpe("", 256, [])
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 0)

    def test_single_character_input(self):
        bpe = Bpe("aaaaa", 2, [])
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 2)
        self.assertIn(b'a', bpe.token_to_int)
        self.assertIn(b'aa', bpe.token_to_int)

    def test_no_repeated_pairs(self):
        bpe = Bpe("abcdefg", 8, [])
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 8)
        # No merges should happen as vocab size is less than unique characters.
        self.assertNotIn(b'fg', bpe.token_to_int)

    def test_vocab_size_equal_to_initial_tokens(self):
        bpe = Bpe("abc", 3, [])
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 3)
        self.assertIn(b'a', bpe.token_to_int)
        self.assertIn(b'b', bpe.token_to_int)
        self.assertIn(b'c', bpe.token_to_int)
        self.assertNotIn(b'ab', bpe.token_to_int)

    def test_vocab_size_smaller_than_initial_tokens(self):
        bpe = Bpe("abcde", 6, [])
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 6)

    def test_multiple_special_tokens(self):
        bpe = Bpe("a<|endoftext|>b<|padding|>", 5, ["<|endoftext|>", "<|padding|>"])
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 5)
        self.assertIn(b'<|endoftext|>', bpe.token_to_int)
        self.assertIn(b'<|padding|>', bpe.token_to_int)
        self.assertIn(b'ab', bpe.token_to_int)

    def test_special_token_as_substring(self):
        bpe = Bpe("a<|endoftext|>b", 4, ["<|endoftext|>"])
        bpe.train()
        self.assertEqual(len(bpe.vocab_mapping()), 4)
        self.assertIn(b'<|endoftext|>', bpe.token_to_int)
        self.assertIn(b'ab', bpe.token_to_int)

if __name__ == '__main__':
    unittest.main()
