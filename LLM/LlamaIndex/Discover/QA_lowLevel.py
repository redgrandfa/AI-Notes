from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine

retriever = VectorIndexRetriever(
  index = index,
  similarity_top_k = 3
)

synth = get_response_synthesizer(
  response_mode = "accumulate"
)

query_engine = RetrieverQueryEngine(
  retriever = retriever,
  response_synthesizer = synth,
)

response = query_engine.query("what is ...")

# response.source_nodes
