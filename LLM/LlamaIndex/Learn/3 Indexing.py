# an Index is a data structure composed of Document objects, designed to enable querying by an LLM.
# complementary to your querying strategy

'''
A VectorStoreIndex is the most frequent type
 Vector Store Index takes your Documents and splits them up into Nodes. 
creates vector embeddings of every node

A vector embedding, often just called an embedding, 
is a numerical representation of the semantics, or meaning of your text.

This mathematical relationship enables semantic search, 

By default [text-embedding-ada-002], 
which is the default embedding used by OpenAI. 
use different LLMs => will often want to use different embeddings.

your query is itself turned into a vector embedding
VectorStoreIndex rank all the embeddings by how semantically similar they are to your query.

---Top K semantic  Retrieval
VectorStoreIndex returns the most-similar embeddings as their corresponding chunks of text
'''


# from_documents optional argument 
#   show_progress=True. # display a progress bar during index construction.

from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)
index = VectorStoreIndex(nodes) # over a list of Node objects directly

'''
# embedding => time-consuming and expensive if you are using a hosted LLM 
# =>  store your embeddings first.
'''

'''
Summary Index
a simpler form of Index 
best suited to queries you are trying to generate a summary. 
It simply [stores all of the Documents and returns all of them].
'''

'''graph index
If your data is a set of interconnected concepts (in computer science terms, a "graph") 
'''