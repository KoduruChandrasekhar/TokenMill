"""Incremental detokenization with stop strings.

Decoding one token at a time breaks characters that span several tokens (emoji, Hindi, Chinese):
a byte-level BPE token can hold half of a UTF-8 character. So we keep all output ids, decode a
small tail window starting at prefix_offset, and emit only text that is new and complete. Text
ending in U+FFFD (the replacement character) is held back until the next token completes it.

Stop strings are matched against the decoded text. While stop strings are active, the last
(longest stop - 1) characters are held back so a stop string is never half-streamed; the stop
string itself is trimmed from the final text.
"""

from __future__ import annotations


class IncrementalDetokenizer:
    def __init__(self, tokenizer, stop: list[str] | None = None, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.stop = [s for s in (stop or []) if s]
        self.hold = max((len(s) for s in self.stop), default=1) - 1
        self.skip_special = skip_special_tokens
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0
        self.text = ""             # all decoded text (complete characters only)
        self.sent = 0              # characters already handed out
        self.stopped = False       # a stop string matched; text is final
        self.num_tokens_at_stop: int | None = None

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=self.skip_special)

    def add(self, token_ids: list[int]) -> str:
        """Feed new token ids; return the text that may be streamed now."""
        if self.stopped:
            return ""
        for tok in token_ids:
            self.ids.append(tok)
            prefix_text = self._decode(self.ids[self.prefix_offset:self.read_offset])
            new_text = self._decode(self.ids[self.prefix_offset:])
            if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
                delta = new_text[len(prefix_text):]
                self.prefix_offset = self.read_offset
                self.read_offset = len(self.ids)
                start = max(0, len(self.text) - self.hold)
                self.text += delta
                if self._check_stop(start):
                    self.num_tokens_at_stop = len(self.ids)
                    break
        if self.stopped:
            return self._take(len(self.text))
        return self._take(len(self.text) - self.hold if self.stop else len(self.text))

    def _check_stop(self, search_from: int) -> bool:
        best = -1
        for s in self.stop:
            i = self.text.find(s, search_from)
            if i != -1 and (best == -1 or i < best):
                best = i
        if best == -1:
            return False
        self.text = self.text[:best]
        self.stopped = True
        return True

    def _take(self, upto: int) -> str:
        upto = max(upto, self.sent)
        out = self.text[self.sent:upto]
        self.sent = upto
        return out

    def flush(self) -> str:
        """End of generation: release everything held back, including a trailing incomplete
        character (as U+FFFD), so the full text equals tokenizer.decode(all ids)."""
        if not self.stopped and self.read_offset < len(self.ids):
            prefix_text = self._decode(self.ids[self.prefix_offset:self.read_offset])
            new_text = self._decode(self.ids[self.prefix_offset:])
            if len(new_text) > len(prefix_text):
                start = max(0, len(self.text) - self.hold)
                self.text += new_text[len(prefix_text):]
                if self._check_stop(start):
                    self.num_tokens_at_stop = len(self.ids)
            self.prefix_offset = self.read_offset = len(self.ids)
        return self._take(len(self.text))
