'''Persisting to disk

'''
index.storage_context.persist(persist_dir="<persist_dir>")

# Here is an example of a Composable Graph:
graph.root_index.storage_context.persist(persist_dir="<persist_dir>")


# loading the persisted index 
from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")

# load index
index = load_index_from_storage(storage_context)
# Important: if you had initialized your index with a custom transformations, embed_model, etc., you will need to pass in the same options 
# during load_index_from_storage


'''Using Vector Stores
The API calls to create the {ref}embeddings <what-is-an-embedding>???
'''
# Chroma:
# pip install chromadb

To use  to store the embeddings from a VectorStoreIndex, you need to:
  - initialize the Chroma client
  - create a Collection to store your data in Chroma
  - assign Chroma as the vector_store in a StorageContext
  - initialize your VectorStoreIndex using that StorageContext

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# if 首次
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context
)

# if 已有
index = VectorStoreIndex.from_vector_store(
    vector_store, 
    storage_context=storage_context
)


'''Inserting Documents or Nodes#
'''
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex([])
for doc in documents:
    index.insert(doc)