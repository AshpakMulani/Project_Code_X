import streamlit as st
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from utils import source_import, vector_store, embedding, llm_openai, support_functions



index_name = 'osho-index'


load_dotenv()


@dataclass
class Message:
    actor: str
    payload: str


USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hello! How can I help you?")]

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    
    response = support_functions.get_response(index_name,prompt)

    print(response)

    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response) 







with st.sidebar:
        st.subheader("Source Documents")
        pdf_docs = st.file_uploader(
            "Upload source PDFs here and click on 'Create Embedding' for storing embeddings in Vector store", accept_multiple_files=True)
        
        
        if st.button("Create Embedings"):
            with st.spinner("Processing"):

                try:
                    text = source_import.get_pdf_text(pdf_docs)
                    chunks = source_import.get_pdf_chunks(text)

                    embedding_list = embedding.create_embeddings_list(chunks)

                    v_index=vector_store.get_Index(index_name)

                    results = vector_store.upsert_embeddings(v_index,embedding_list,'dummy path')

                    st.success("Done!")
                except:
                    st.error("Error occured..!!")
                

   

