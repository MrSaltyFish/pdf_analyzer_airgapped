from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # tokens or characters based on tokenizer (depends on use)
    chunk_overlap=100,    # to preserve context
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)
