from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()



from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)



from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")


# 重點:
from llama_index.core import SummaryIndex, VectorStoreIndex

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

query_engine = RouterQueryEngine(
  selector=LLMSingleSelector.from_defaults(),
  query_engine_tools=[
    summary_tool,
    vector_tool,
  ],
  verbose=True
)

response = query_engine.query("What is the summary of the document?")
print(str(response))

print(len(response.source_nodes)) #看理由  summary一定是全部nodes

response = query_engine.query(
    "How do agents share information with other agents?"
)
print(str(response))