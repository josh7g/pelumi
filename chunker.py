import tiktoken

def chunk_source_code(source_code: str, max_tokens: int = 7692, overlap: int = 10) -> list:
    """
    Splits source code into chunks of less than max_tokens while preserving line integrity.
    :param source_code: The source code as a string.
    :param max_tokens: The maximum number of tokens per chunk.
    :param overlap: The number of lines to overlap between chunks.
    :return: A list of code chunks.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Adjust based on your tokenizer
    lines = source_code.split('\n')
    chunks = []
    current_chunk = []
    current_token_count = 0

    for i, line in enumerate(lines):
        token_count = len(tokenizer.encode(line))

        # If adding this line exceeds max tokens, finalize the current chunk
        if current_token_count + token_count > max_tokens:
            chunks.append('\n'.join(current_chunk))  # Convert list of lines to a single string

            # Start new chunk with overlap
            current_chunk = lines[max(0, i - overlap):i + 1]  # Include last `overlap` lines
            current_token_count = sum(len(tokenizer.encode(l)) for l in current_chunk)
        else:
            current_chunk.append(line)
            current_token_count += token_count

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks
