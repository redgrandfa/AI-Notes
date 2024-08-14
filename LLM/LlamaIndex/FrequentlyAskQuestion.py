# ==== High Level 五行
from llama_index.core import Settings

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)

# ======== 1  Loading Stage
# A data connector (often called a Reader) ingests data from different data sources and data formats into Documents and Nodes.

documents = SimpleDirectoryReader("data").load_data()  #documents是一個list

# ======== 2  Indexing Stage
  # index the data into a structure that's easy to retrieve. 
  # This usually involves generating vector embeddings 
  #   which are stored in a specialized database called a vector store. 
  # Indexes can also store a variety of metadata about your data.

  # Embeddings 
    # LLMs generate numerical representations of data called embeddings.  
    # When filtering your data for relevance, LlamaIndex will convert queries into embeddings, 
    # [vector store will find data] that is [numerically similar to the embedding of your query].

# ==== want to parse my documents into smaller chunks
  # # Global settings
  # Settings.chunk_size = 512

  # # Local settings
  # from llama_index.core.node_parser import SentenceSplitter
  # index = VectorStoreIndex.from_documents(
  #     documents, transformations=[SentenceSplitter(chunk_size=512)]
  # )

# ==== want to use a different vector store
  # For example, to use Chroma as the vector store
  # pip install llama-index-vector-stores-chroma
  # about all integrations available, check out LlamaHub.

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

chroma_client = chromadb.PersistentClient()
chroma_collection = chroma_client.create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# StorageContext defines the storage backend  ( about storage)
index = VectorStoreIndex.from_documents(documents, 
  storage_context=storage_context
)

# ======== 3  Querying Stage
# Retrievers: 
  # retrieve relevant context from an index when given a query. 
  # relevancy , efficiency 

# Routers: 
  # A router determines which retriever will be used to retrieve relevant context from the knowledge base. 
  # More specifically, the RouterRetriever class, is responsible for selecting one or multiple candidate retrievers to execute a query. 
  # They use a [selector] to choose the best option based on each candidate's metadata and the query.

# Node Postprocessors: 
#   A node postprocessor takes in a set of retrieved nodes and applies transformations, filtering, or re-ranking logic to them.

# Response Synthesizers: 
  # (given: query , set of retrieved text chunks.)
  # A response synthesizer generates a response from an LLM, 



# ==== want to retrieve more context when I query
  # as_query_engine builds a [default retriever] and [query engine] on top of the index. 
  # You can configure the retriever and query engine by passing in keyword arguments. 
  # You can learn more about retrievers and query engines.
query_engine = index.as_query_engine(
  similarity_top_k=5, #default of 2
  response_mode="tree_summarize" # ==== want to use a different response mode
) 

# ==== want to use a different LLM
#   # Global settings
#   from llama_index.core import Settings
#   from llama_index.llms.ollama import Ollama

#   Settings.llm = Ollama(model="mistral", request_timeout=60.0)

#   # Local settings
#   index.as_query_engine(llm=Ollama(model="mistral", request_timeout=60.0))



response = query_engine.query("What did the author do growing up?")
print(response) 

# ==== want to stream the response back
#   query_engine = index.as_query_engine(streaming=True)
#   response = query_engine.query(...)
#   response.print_response_stream()
#   #learn more about streaming responses.


# ==== want a chatbot instead of Q&A
#   query_engine = index.as_chat_engine()
#   response = query_engine.chat("What did the author do growing up?")
#   print(response)

#   response = query_engine.chat("Oh interesting, tell me more.")
#   print(response)


# ======== 4  EVALUATE


