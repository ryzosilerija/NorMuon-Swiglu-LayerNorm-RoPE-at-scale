"""Byte-level BPE tokenizer, written from scratch (the kind GPT-2/GPT-4 use).

How it works
  * Text is turned into UTF-8 bytes, so every possible character (any
    language, emoji) can be represented: the base vocabulary is 256 bytes.
  * Before merging, text is split into chunks with a regex (words, numbers,
    punctuation, whitespace), so merges never cross word boundaries: you get
    tokens like " the" or "ing", never "e. T".
  * Training repeatedly merges the most frequent pair of neighbouring tokens
    into a new token until the vocabulary reaches the target size. The ordered
    list of merges *is* the tokenizer.
  * Encoding applies the merges to new text in the order they were learned;
    decoding just concatenates each token's bytes.
  * Special tokens (<|user|>, <|assistant|>, ...) get their own ids and are
    never split. They are added later for chat (step 4) but reserved now.

Training is fast because it works on *unique chunks with counts* (a word that
appears 50,000 times is processed once) and only updates the pair counts of
chunks that contain the pair being merged.
"""
import json
from collections import Counter, defaultdict

import regex

# GPT-4's split pattern: contractions, letters (with an optional leading non-letter like a space),
# numbers of up to 3 digits, punctuation runs, newlines, other whitespace
SPLIT_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
                 r"""|\s*[\r\n]|\s+(?!\S)|\s+""")
DEFAULT_SPECIAL = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


class BPETokenizer:
    def __init__(self):
        self.merges = {}            # (id_a, id_b) -> new id, in the order learned
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special = {}           # "<|user|>" -> id
        self.pattern = regex.compile(SPLIT_PATTERN)
        self._cache = {}

    # ------------------------------------------------------------------ training
    def train(self, text, vocab_size, special_tokens=DEFAULT_SPECIAL, verbose=True):
        n_merges = vocab_size - 256 - len(special_tokens)
        assert n_merges > 0, "vocab_size too small"
        chunk_counts = Counter(self.pattern.findall(text))
        words = [list(c.encode("utf-8")) for c in chunk_counts]   # each unique chunk as a list of ids
        freqs = list(chunk_counts.values())

        pair_counts = Counter()
        where = defaultdict(set)    # pair -> indices of words containing it
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] += freqs[wi]
                where[p].add(wi)

        for k in range(n_merges):
            if not pair_counts:
                break
            pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))  # deterministic ties
            if pair_counts[pair] < 2:
                break
            new_id = 256 + k
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            for wi in list(where[pair]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):                 # remove this word's old pairs
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    where[p].discard(wi)
                words[wi] = w = _merge(w, pair, new_id)
                for p in zip(w, w[1:]):                 # add its new pairs
                    pair_counts[p] += f
                    where[p].add(wi)
            if verbose and (k + 1) % 500 == 0:
                print(f"  merge {k + 1}/{n_merges}: {self.vocab[new_id]!r}")

        next_id = 256 + len(self.merges)
        self.special = {s: next_id + i for i, s in enumerate(special_tokens)}
        self._cache = {}
        return self

    # ------------------------------------------------------------------ encoding
    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special)

    def _encode_chunk(self, chunk):
        ids = self._cache.get(chunk)
        if ids is not None:
            return ids
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # the pair that was learned earliest gets merged first
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text, allow_special=True):
        """Text -> ids. Special tokens written in the text are recognized only if allow_special
        (never allow it for untrusted user text in a chat app: a user could fake '<|assistant|>')."""
        if not (allow_special and self.special):
            return self._encode_ordinary(text)
        pat = "(" + "|".join(regex.escape(s) for s in self.special) + ")"
        out = []
        for part in regex.split(pat, text):
            if part in self.special:
                out.append(self.special[part])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def _encode_ordinary(self, text):
        out = []
        for chunk in self.pattern.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        inv_special = {i: s.encode("utf-8") for s, i in self.special.items()}
        data = b"".join(self.vocab[i] if i in self.vocab else inv_special[i] for i in ids)
        return data.decode("utf-8", errors="replace")   # a cut-off multi-byte character shows as �

    def token_str(self, i):
        for s, j in self.special.items():
            if j == i:
                return s
        return self.vocab[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ save / load
    def save(self, path):
        json.dump({"pattern": SPLIT_PATTERN, "merges": [[a, b] for (a, b) in self.merges],
                   "special": self.special}, open(path, "w", encoding="utf-8"))

    @classmethod
    def load(cls, path):
        blob = json.load(open(path, encoding="utf-8"))
        tok = cls()
        for k, (a, b) in enumerate(blob["merges"]):
            tok.merges[(a, b)] = 256 + k
            tok.vocab[256 + k] = tok.vocab[a] + tok.vocab[b]
        tok.special = {s: int(i) for s, i in blob["special"].items()}
        return tok


def _merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in ids with new_id."""
    out, i, a, b = [], 0, pair[0], pair[1]
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
