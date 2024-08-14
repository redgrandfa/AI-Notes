# v Viewing Queries and Events Using Logging
# import logging
# import sys

# #set the level to DEBUG for verbose output, or use level=logging.INFO for less
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) 
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# # ^ Viewing Queries and Events Using Logging



# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents)

import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# store the index if it doesn't exist, but load it if it does:
PERSIST_DIR = "./storage/starter"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()  #Documents是一個list。
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    #By default, the data you just loaded is stored in memory as a series of vector embeddings. 
    #You can save time (and requests to OpenAI) by saving the embeddings to disk.

else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)






query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response) #The author wrote short stories and tried to program on an IBM 1401.





