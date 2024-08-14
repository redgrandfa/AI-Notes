'''
which LLM to use; you can also use more than one if you wish.

During Indexing you may use an LLM to determine the relevance of data (whether to index it at all) 
or you may use an LLM to summarize the raw data and index the summaries instead.

During Querying LLMs can be used in two ways:
  During Retrieval (fetching data from your index) 
    LLMs can be given an array of options (such as multiple different indices) and make decisions about where best to find the information you're looking for. 
  An agentic LLM can also use tools at this stage to query different data sources.

  During Response Synthesis (turning the retrieved data into an answer) 
    an LLM can combine answers to multiple sub-queries into a single coherent answer, or it can transform data, such as from unstructured text to JSON or another programmatic output format.


LlamaIndex provides a single interface to a large number of different LLMs, 
allowing you to pass in any LLM you choose to any stage of the pipeline. 

It can be as simple as this:
'''
from llama_index.llms.openai import OpenAI

response = OpenAI().complete("Paul Graham is ")
print(response)


'''
Usually, you will instantiate an LLM and pass it to Settings,
which you then pass to other stages of the pipeline, as in this example:
'''
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

#  instantiated OpenAI and customized it to use the gpt-4 model instead of the default gpt-3.5-turbo, and also modified the temperature. 
Settings.llm = OpenAI(temperature=0.2, model="gpt-4")

# The VectorStoreIndex will now use gpt-4 to answer questions when querying.
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
)

# Available LLMs
# Using a local LLM

# Prompts
# By default LlamaIndex comes with a great set of built-in, battle-tested prompts that handle the tricky work of getting a specific LLM to correctly handle and format data. This is one of the biggest benefits of using LlamaIndex. 
# If you want to, you can customize the prompts
#  models/prompts
