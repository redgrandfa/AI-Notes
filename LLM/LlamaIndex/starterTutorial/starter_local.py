# v Viewing Queries and Events Using Logging
# import logging
# import sys

# #set the level to DEBUG for verbose output, or use level=logging.INFO for less
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) 
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# # ^ Viewing Queries and Events Using Logging




# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# documents = SimpleDirectoryReader("data").load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# index = VectorStoreIndex.from_documents(
#     documents,
# )


# view logs, persist/load the index similar to our starter example.


import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
# store the index if it doesn't exist, but load it if it does:
PERSIST_DIR = "./storage/starter_local"
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


# =========
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

# To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator: 在 Windows 上支持符號鏈接，你需要啟用開發者模式或者以管理員身份運行 Python。
# In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development: 要啟用開發者模式，請參閱這篇文章。


# raise HTTPStatusError(message, request=request, response=self)
# httpx.HTTPStatusError: Server error '502 Bad Gateway' for url 'http://localhost:11434/api/chat'
# For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/502