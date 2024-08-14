'''Evaluation and benchmarking 
To improve the performance of an LLM ap

LlamaIndex offers 
  key modules to measure the [quality of generated results]. 
  key modules to measure [retrieval quality]. 
You can learn more about how evaluation works in LlamaIndex in our module guides.
'''

'''Response Evaluation#
Does the response match the retrieved context? 
Does it also match the query? 
Does it match the reference answer or guidelines? 

Here's a simple example that evaluates a single response for Faithfulness, 
  i.e. whether the response is aligned to the context, 
    such as being free from hallucinations:

'''
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator

# create llm
llm = OpenAI(model="gpt-4", temperature=0.0)

# build index
...
vector_index = VectorStoreIndex(...)

# define evaluator
evaluator = FaithfulnessEvaluator(llm=llm)

# query index
query_engine = vector_index.as_query_engine()
response = query_engine.query("...")
eval_result = evaluator.evaluate_response(response=response)
print(str(eval_result.passing))
# The response contains 
#   response 
#   the source from which the response was generated;
# the evaluator compares them

# You can learn more in our module guides about response evaluation.

'''Retrieval Evaluation#
'''
from llama_index.core.evaluation import RetrieverEvaluator

# define retriever somewhere (e.g. from index)
# retriever = index.as_retriever(similarity_top_k=2)
retriever = ...

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

retriever_evaluator.evaluate(
    query="query", 
    expected_ids=["node_id1", "node_id2"]
)
# In reality you would want to evaluate a whole batch of retrievals; 
# you can learn how do this in our module guide on retrieval evaluation.

'''Related concepts#
[analyzing the cost of your application] if you are making calls to a hosted, remote LLM.
'''

