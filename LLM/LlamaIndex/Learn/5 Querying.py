query_engine = index.as_query_engine()
response = query_engine.query(
    "Write an email to the user given their background information."
)
'''Stages of querying#
Retrieval 
  the most relevant documents for your query from your Index. 
  (the most common type of retrieval is "top-k" semantic retrieval, but there are many other retrieval strategies.)

Postprocessing 
  is when the Nodes retrieved are optionally [reranked, transformed, or filtered], 
  for instance by requiring that they have specific metadata such as keywords attached.

Response synthesis 
  most-relevant data and  prompt are combined and sent to LLM to return a response.
'''
#  find out about how to attach metadata to documents and nodes.

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

index = VectorStoreIndex.from_documents(documents)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)], 
    #設下限
)


'''Configuring retriever#
'''
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)


'''Configuring node postprocessors#
We support advanced Node filtering and augmentation that can further improve the relevancy of the retrieved Node objects. 
This can help reduce the time/number of LLM calls/cost or improve response quality.

For example:

KeywordNodePostprocessor: 
  filters nodes 
    required_keywords 
    exclude_keywords.
SimilarityPostprocessor: 
  filters nodes 
    threshold on the similarity score (only supported by embedding-based retrievers)
PrevNextNodePostprocessor: 
  augments retrieved Node objects with additional relevant context based on Node relationships.
'''
node_postprocessors = [
    KeywordNodePostprocessor(
        required_keywords=["Combinator"], exclude_keywords=["Italy"]
    ),
    SimilarityPostprocessor(similarity_cutoff=0.7)
]
query_engine = RetrieverQueryEngine.from_args(
    retriever, 
    node_postprocessors=node_postprocessors
)

'''Configuring response synthesis#
a [BaseSynthesizer] synthesizes the final response by combining the information.
'''
# configure response synthesis
query_engine = RetrieverQueryEngine.from_args(
    retriever, 
    response_mode=response_mode
)

'''we support the following options:

default: 
  "create and refine" an answer by sequentially going through each retrieved Node; This makes a separate LLM call per Node. Good for more detailed answers.

compact: 
  "compact" the prompt during each LLM call by stuffing as many Node text chunks that can fit within the maximum prompt size. If there are too many chunks to stuff in one prompt, "create and refine" an answer by going through multiple prompts.

tree_summarize: 
  Given a set of Node objects and the query, recursively construct a tree and return the root node as the response. Good for summarization purposes.

no_text: 
  Only runs the retriever to fetch the nodes that would have been sent to the LLM, without actually sending them. Then can be inspected by checking response.source_nodes. The response object is covered in more detail in Section 5.

accumulate: 
  Given a set of Node objects and the query, apply the query to each Node text chunk while accumulating the responses into an array. Returns a concatenated string of all responses. Good for when you need to run the same query separately against each text chunk.
'''


'''Structured Outputs#
You may want to ensure your output is structured. 
See our Query Engines + Pydantic Outputs to see how to extract a Pydantic object from a query engine class.

Also make sure to check out our entire Structured Outputs guide.
'''

'''Creating your own Query Pipeline
If you want to design complex query flows, ...
'''