# Define prompt template
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain.load import dumps, loads
import json
#from langchain_openai import OpenAI
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import yaml
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone
from operator import itemgetter
from langchain_pinecone import PineconeVectorStore





load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAPI_KEY')

def get_standalone_response_lc(query,template_id):

    prompt_type='standalone-prompts'
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vector_index='osho-index'
    vector_namespace = "OSHO-namespace"

    prompt_template = str(read_prompt(prompt_type,template_id))
     
    prompt_perspectives = ChatPromptTemplate.from_template(prompt_template)

    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY) 
        | (lambda x: x.split("\n"))
    )

    pc_interface = PineconeVectorStore.from_existing_index(
                vector_index, 
                embedding=embed_model, 
                namespace=vector_namespace
                )
    
    retriever=pc_interface.as_retriever()

    print(generate_queries)

    generate_queries.invoke({"question":query})


    retrieval_chain = generate_queries | retriever.map() | get_unique_union

    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0)

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    final_rag_chain.invoke({"question":query})
        

def get_standalone_response(query, template_id, context='', query_type=None):

    prompt_type='standalone-prompts'


    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt_template = str(read_prompt(prompt_type,template_id))
    prompt_template = prompt_template.replace("{question}",query)
    prompt_template = prompt_template.replace("{context}",context)


    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are helpful assistant"},
                            {"role": "user", "content": prompt_template}
                        ])
    
    response = completion.choices[0].message.content

    if query_type == 'multi-query':
        response = str(completion.choices[0].message.content).split('\n')

        print("========= multi query results=======\n")
        print(*response, sep="\n")
        
    return response




def get_conversational_response(query, template_id):

    prompt_type='conversation-prompts'

    ## currenly it looks like there wont be need for this whihc included
    ## conversation chain using biffer memory because we need to figure
    ## out how chain will be used when microservices archotecture is used
    ## like lambda to interact with llm on each user query
    ## for now it seems conversaiton history can be managed by client
    ## and histiry summary can be passed inside context along with query

    return None


def read_prompt(type,id):
        parent_parent_dir = os.path.join(os.path.dirname(__file__), '..')

        with open(os.path.join(parent_parent_dir, 'config\prompt_templates.yml'), 'r') as file:
            yaml_content = yaml.safe_load(file)

        return yaml_content[type][id]


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [json.dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [json.loads(doc) for doc in unique_docs]








