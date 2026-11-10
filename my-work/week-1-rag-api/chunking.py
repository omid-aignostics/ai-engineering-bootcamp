"""Session 2 — split long text into overlapping chunks before it goes into the vector store."""

CHUNK_SIZE = 1000  # Characters per chunk: small enough to be specific, big enough to keep context.
CHUNK_OVERLAP = 150  # Characters repeated between neighbours, so a sentence cut at a boundary survives.


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on word boundaries into chunks of at most ~chunk_size characters."""

    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end, length = start, 0
        # Always take at least one word, even one longer than chunk_size.
        while end < len(words) and (end == start or length + len(words[end]) + 1 <= chunk_size):
            length += len(words[end]) + 1
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break

        # Start the next chunk a few words back, so it repeats about `overlap` characters.
        back, repeated = end, 0
        while back > start + 1 and repeated < overlap:
            back -= 1
            repeated += len(words[back]) + 1
        start = back
    return chunks
