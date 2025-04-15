from tree_sitter import Language, Parser
#import tree_sitter.tree_sitter_python
import tree_sitter_python
import tiktoken

PY_LANGUAGE = Language(tree_sitter_python.language())

# Initialize tokenizer (Titan supports OpenAI's tiktoken)
tokenizer = tiktoken.get_encoding("cl100k_base")  # Approximate tokenizer

TOKEN_LIMIT = 7000  # Amazon Titan's context limit

def count_tokens(text):
    """Estimate token count using tiktoken."""
    return len(tokenizer.encode(text))

def extract_chunks(source_code):
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(source_code.encode("utf8"))
    root_node = tree.root_node

    chunks = []
    current_chunk = ""
    current_chunk_size = 0
    #chunk_index = 1

    def add_chunk(text):
        """Helper to add a chunk if it doesn't exceed token limit."""
        nonlocal current_chunk, current_chunk_size
        token_count = count_tokens(text)

        if current_chunk_size + token_count > TOKEN_LIMIT:
            # Store previous chunk and start a new one
            chunks.append(current_chunk.strip()) #Append previous chunk
            current_chunk = text
            current_chunk_size = token_count
        else:
            current_chunk += "\n\n" + text
            current_chunk_size += token_count

    # Process classes
    for node in root_node.children: # Loops through the top-level nodes of the parsed syntax tree.
        if node.type == "class_definition":
            class_body = source_code[node.start_byte:node.end_byte] # Extracts the entire class definition from source_code using byte offsets.
            if count_tokens(class_body) <= TOKEN_LIMIT:
                add_chunk(class_body)
            else:
                # Class is too large, split into methods but extract only the class header (without its body)
                class_header = source_code[node.start_byte:node.child_by_field_name("body").start_byte]
                add_chunk(class_header)

                # Extracts individual methods inside the class and adds them separately as chunks.
                class_body_node = node.child_by_field_name("body")
                if class_body_node:
                    for func in class_body_node.children:
                        if func.type == "function_definition":
                            func_body = source_code[func.start_byte:func.end_byte]
                            add_chunk(func_body)

    # Process standalone functions
    for node in root_node.children:
        if node.type == "function_definition":
            func_body = source_code[node.start_byte:node.end_byte]
            add_chunk(func_body)

    # Process remaining chunk if not added
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
    #return ("\n\n".join(chunks),)


if __name__ == "__main__":

    # Read testdata.py
    with open("testdata2.py", "r") as file:
        sourcecode = file.read()

    # Extract and print chunks
    chunk_list = extract_chunks(sourcecode)
    # Print chunks
    for idx, chunk in enumerate(chunk_list, start=1):
        print(f"Chunk {idx}:\n{chunk}\n")

    # Return token count
    print(f"{count_tokens(sourcecode)} Total Tokens")

    # Return chunk count
    print(f"{len(chunk_list)} Total Chunks")
