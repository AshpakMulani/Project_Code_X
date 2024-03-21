
from utils import source_import, vector_store, embedding, llm_openai

def get_response(index, query):
    matched_vector_id=[]
    matched_vector_metadata=[]

    if len(query) > 20:
        multi_query_response = llm_openai.get_standalone_response(query,'osho-multi-query','','multi-query')

        if len(multi_query_response)>0:
            for query in multi_query_response:
                matched_vector_id,matched_vector_metadata = get_matching_blocks(
                                                                index,
                                                                query,
                                                                matched_vector_id,
                                                                matched_vector_metadata
                                                            )                
    else:   
        matched_vector_id,matched_vector_metadata = get_matching_blocks(
                                                        index,
                                                        query,
                                                        matched_vector_id,
                                                        matched_vector_metadata
                                                    )
        
    return matched_vector_metadata



def get_matching_blocks(index, query, vector_id=[], vector_metadata=[]):
    query_embedded = embedding.create_embeddings_string(query)

    matching_vectors = vector_store.get_matching_vectors(vector_store.get_Index(index), query_embedded,5)

    if matching_vectors != None and len(matching_vectors['matches'])>0:
        for vector in matching_vectors['matches']:
          if vector['id'] not in vector_id:
            vector_id.append(vector['id'])
            vector_metadata.append(vector['metadata'])

    return [vector_id, vector_metadata]
            