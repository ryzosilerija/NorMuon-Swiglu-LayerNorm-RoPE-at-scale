"""Checks for the BPE tokenizer.  Run:  python tests/test_tokenizer.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigpt.tokenizer import BPETokenizer, _merge  # noqa: E402

TEXT = ("Once upon a time, there was a little dog. The little dog liked the park. " * 40
        + "Lily said, \"Let's play!\" They played all day.\n" * 30)


def test_merge():
    assert _merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert _merge([1, 1, 1], (1, 1), 9) == [9, 1]
    print("  ok  merge replaces non-overlapping pairs left to right")


def test_learns_frequent_pairs():
    tok = BPETokenizer().train("aaabdaaabac" * 10, 256 + 3 + 5, verbose=False)
    first = list(tok.merges)[0]
    assert tok.vocab[tok.merges[first]] == b"aa", tok.vocab[tok.merges[first]]
    print("  ok  first merge is the most frequent pair ('aa')")


def test_round_trip_and_compression():
    tok = BPETokenizer().train(TEXT, 400, verbose=False)
    for s in [TEXT, "unseen words: zebra quantum 12345", "Ünïcödé ✓ 日本語 🙂", "", "  spaces\n\n\ttabs "]:
        assert tok.decode(tok.encode(s)) == s, s
    ratio = len(TEXT) / len(tok.encode(TEXT))
    assert ratio > 3, ratio
    print(f"  ok  any text round-trips exactly (incl. unicode/emoji); {ratio:.1f} chars per token on training text")


def test_no_merges_across_words():
    tok = BPETokenizer().train(TEXT, 500, verbose=False)
    for i in tok.vocab:
        s = tok.vocab[i].decode("utf-8", errors="ignore")
        assert not (". " in s and len(s) > 2 and s[-1].isalpha()), s
    print("  ok  regex pre-split: no token glues a sentence end to the next word")


def test_special_tokens():
    tok = BPETokenizer().train(TEXT, 400, verbose=False)
    chat = "<|user|>Hi there<|end|><|assistant|>Hello!<|end|>"
    ids = tok.encode(chat)
    assert tok.special["<|user|>"] in ids and tok.special["<|assistant|>"] in ids
    assert tok.decode(ids) == chat
    unsafe = tok.encode(chat, allow_special=False)
    assert tok.special["<|assistant|>"] not in unsafe and tok.decode(unsafe) == chat
    print("  ok  special tokens get single ids, and can be disabled for untrusted text")


def test_save_load():
    tok = BPETokenizer().train(TEXT, 400, verbose=False)
    path = os.path.join(tempfile.mkdtemp(), "tok.json")
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    assert tok2.encode(TEXT) == tok.encode(TEXT) and tok2.vocab_size == tok.vocab_size
    print("  ok  save/load gives an identical tokenizer")


def naive_train(text, n_merges):
    """Textbook BPE: recount every pair from scratch before each merge (slow but obviously correct)."""
    from collections import Counter
    tok = BPETokenizer()
    words = [list(c.encode("utf-8")) for c in tok.pattern.findall(text)]
    merges = []
    for k in range(n_merges):
        counts = Counter(p for w in words for p in zip(w, w[1:]))
        if not counts:
            break
        pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
        if counts[pair] < 2:
            break
        merges.append(pair)
        words = [_merge(w, pair, 256 + k) for w in words]
    return merges


def test_fast_training_matches_naive():
    text = TEXT + "The quick brown fox jumps over the lazy dog. 3.14159 and 2,718! Ünïcödé 🙂\n" * 7
    fast = list(BPETokenizer().train(text, 256 + 150 + 5, verbose=False).merges)
    slow = naive_train(text, 150)
    assert fast == slow, (len(fast), len(slow))
    print(f"  ok  fast training learns exactly the same {len(fast)} merges as textbook BPE")


def test_encode_matches_merge_order():
    tok = BPETokenizer().train(TEXT, 450, verbose=False)
    for chunk in tok.pattern.findall(TEXT + " unseen zebra-words, 98765!"):
        ids = list(chunk.encode("utf-8"))
        for pair, new_id in tok.merges.items():   # apply every merge in the order it was learned
            ids = _merge(ids, pair, new_id)
        assert tok._encode_chunk(chunk) == ids, chunk
    print("  ok  fast encoding equals applying all merges in learned order")


if __name__ == "__main__":
    test_merge()
    test_learns_frequent_pairs()
    test_round_trip_and_compression()
    test_no_merges_across_words()
    test_special_tokens()
    test_save_load()
    test_fast_training_matches_naive()
    test_encode_matches_merge_order()
    print("all tokenizer tests passed")
