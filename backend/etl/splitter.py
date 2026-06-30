"""
Text chunker — splits transcripts into overlapping character-bounded chunks.

Default: chunk_size=500, overlap=100, breaks at sentence boundaries when possible.
"""

from __future__ import annotations

from dataclasses import dataclass

_BREAK_SEPS = [". ", "? ", "! ", "\n\n", "\n", "; ", " "]


@dataclass
class Chunk:
    text: str
    index: int
    char_start: int  # byte offset in the full original text
    char_end: int


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """
    Split text into overlapping chunks of up to chunk_size characters.

    Boundaries are snapped to the nearest sentence/word ending to avoid
    cutting mid-sentence. char_start / char_end reference the original text,
    enabling downstream timestamp mapping from Whisper segments.
    """
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Snap backward to a natural sentence/word boundary
            for sep in _BREAK_SEPS:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    char_start=start,
                    char_end=end,
                )
            )

        # Next chunk starts overlap characters before end of this chunk
        start = max(start + 1, end - overlap)

    return chunks
