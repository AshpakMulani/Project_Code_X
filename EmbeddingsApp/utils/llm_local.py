import ollama



# Define prompt template

multi_query_prompt_template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {question}"""


gen_prompt_template ="""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {query}
"""



def get_multi_query_prompt(query):
    return multi_query_prompt_template.replace("{question}",query)


def get_gen_prompt(query, context):
    temp = gen_prompt_template.replace("{query}",query)

    return temp.replace("{context}", context)

def generate_llm_response(prompt):
    response = ollama.chat(model='mistral', messages=[
    {
        "role": "system",
        "content": "You are an expert at assisting with given prompt"
    },
    {
        "role": "user",
        "content": prompt
    },
    ])

    return response


