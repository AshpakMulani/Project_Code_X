
import time
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
import os
import uuid


load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
env='us-west1-gcp'
vector_namespace = "OSHO-namespace"


pc = Pinecone(api_key=PINECONE_API_KEY)


def get_Index(name='default_index'):
 
    if name not in pc.list_indexes().names():
        #create index to store vectors created during the embedding process
        pc.create_index(
            name=name, 
            dimension=384, # vector dimention from OpenAI ada-002 embedding model
            metric="cosine", # cosine similarity metrics for vector search
            spec=PodSpec(environment="us-west1-gcp", pod_type="p1.x1")
        )

        # wait for index to be initialized
        while not pc.describe_index(name).status['ready']:
            time.sleep(1)

    # connect to newely created index
    index = pc.Index(name)

    return index



# Define a function to upsert embeddings to Pinecone
def upsert_embeddings (index, embeddings, file_path):
    for embedding in  embeddings:
        vector = embedding[0]
        vector_text=embedding[1]

        index.upsert(
            vectors=[
            {
            'id':str(uuid.uuid4()), 
            'values':vector, 
            'metadata':{'file': file_path, 'text':vector_text  }
            }
            ],
            namespace='OSHO-namespace'
        )


def get_matching_vectors(index, query_vector, top_k):
    
    matching_vectors = index.query(
        vector=query_vector ,
        top_k=top_k,
        include_metadata=True,
        namespace=vector_namespace
    )

    return matching_vectors

def get_matching_chunk_text(matching_vectors):

    result_text = ''

    for chunk in matching_vectors['matches']:
        result_text = result_text +  "\n\n----------------------\n\n" + chunk['metadata']['text']

    return result_text