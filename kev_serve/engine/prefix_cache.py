"""State-prefix cache: the prefixes (KV + recurrent states) kept across requests, LRU first.

Ported from kev/serve.py's PrefixCache; the only change is that the cache key is computed by the
backend (Backend.prefix_key) instead of read off kev's encoding, so plan() takes precomputed
(key, n_tokens) pairs. The LRU/bounds/eviction semantics are unchanged — a batch keeps (copies) only
the new states that will still be here after it, its last distinct ones within both bounds: the rest
would be evicted by the batch itself."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrefixCache:
    """State prefixes kept across requests, least recently used first: key -> prefix. At most `size`
    states and `max_tokens` state tokens in all; states shorter than min_tokens or longer than
    max_tokens are not cached. A key is whatever the backend's prefix_key returned (usually the
    state's token ids); the prefix is opaque to this class."""
    size: int
    min_tokens: int
    max_tokens: int = 65536
    entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    oom_retries: int = 0   # batches that ran out of device memory with states cached, dropped them and ran again

    def plan(self, items):
        """-> (key per request, None when its state is not cached; its cached prefix or None; whether to
        keep a new one). `items` is one (key, n_tokens) per request, the key None when the backend does
        not cache that request."""
        keys = [key if (self.size and key is not None and self.min_tokens <= n <= self.max_tokens) else None
                for key, n in items]
        survivors, tokens = set(), 0
        for key in dict.fromkeys(k for k in reversed(keys) if k is not None):   # most recent first, as store() keeps them
            if len(survivors) == self.size or tokens + len(key) > self.max_tokens: break
            survivors.add(key); tokens += len(key)
        return keys, [self.entries.get(k) if k is not None else None for k in keys], [k in survivors for k in keys]

    def over(self, keys):
        """Whether these keys (states) exceed either bound: more than `size` states or `max_tokens` tokens."""
        return len(keys) > self.size or sum(len(k) for k in keys) > self.max_tokens

    def make_room(self, keys, cached, keep):
        """Drop now, before the batch runs, the entries this batch's store() will evict anyway, so an old
        state does not stay resident through the passes of the new one that replaces it. store()
        (re)inserts every hit and every kept new state as most recent and evicts the least recently used
        while over a bound; evicting the same keys first leaves the same cache. A hit dropped here is
        still held by `cached` for this batch."""
        order = dict.fromkeys(self.entries)                     # least recently used first, as store() evicts
        for key, old, k in zip(keys, cached, keep):
            if key is not None and (old is not None or k):
                order.pop(key, None); order[key] = None
        while self.over(order):
            key = next(iter(order)); del order[key]; self.entries.pop(key, None)

    def store(self, keys, cached, prefixes):
        """Record hits and misses, and (re)insert the batch's prefixes in order: most recently used last."""
        for key, old, new in zip(keys, cached, prefixes):
            if key is None: continue
            self.hits += old is not None; self.misses += old is None
            if new is None: continue
            self.entries.pop(key, None); self.entries[key] = new
            while self.over(self.entries): self.entries.pop(next(iter(self.entries)))

    def clear(self):
        self.entries.clear()

    def stats(self) -> dict:
        """The /v1/models card's prefix_cache block."""
        return {"size": self.size, "min_state_tokens": self.min_tokens, "max_tokens": self.max_tokens,
                "hits": self.hits, "misses": self.misses, "cached_states": len(self.entries),
                "oom_retries": self.oom_retries}
