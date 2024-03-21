from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Define a function to create embeddings
def create_embeddings_list(texts):
    embeddings_list = []
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    for text in texts:
        embd = embed_model.get_text_embedding(text)
        embd_tpl = (embd, text)
        embeddings_list.append(embd_tpl)
    return embeddings_list



# Define a function to create embeddings
def create_embeddings_string(texts):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    res = embed_model.get_text_embedding(texts)

    return res

