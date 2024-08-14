'''Cost Analysis#
  Concept#
    Each call to an LLM will cost some amount of money 
    The cost of building an index and querying depends on
    - the type of LLM used
    - the type of data structure used
    - parameters used during building
    - parameters used during querying
'''
The cost of building and querying each index is a TODO in the reference documentation. 
In the meantime, we provide the following information:

A high-level overview of the cost structure of the indices.
A token predictor that you can use directly within LlamaIndex!

'''Overview of Cost Structure#
Indices don't require LLM calls at all during building (0 cost):
  - SummaryIndex
  - SimpleKeywordTableIndex - 
      uses a [regex keyword extractor] to extract keywords from each document
  - RAKEKeywordTableIndex - 
      uses a RAKE keyword extractor to extract keywords from each document

Indices require LLM calls during build time
  - TreeIndex - use LLM to hierarchically summarize the text to build the tree
  - KeywordTableIndex - use LLM to extract keywords from each document
'''


'''Query Time#
There will always be >= 1 LLM call during query time, in order to synthesize the final answer. Some indices contain cost tradeoffs between index building and querying. 
  ex.
  SummaryIndex, 
    free to build, 
    query (without filtering or embedding lookups), will call the LLM {math}N times.

Here are some notes regarding each of the indices:

  - SummaryIndex: 
      by default requires {math}N LLM calls, where N is the number of nodes.

  - TreeIndex: 
      by default requires {math}\log (N) LLM calls, where N is the number of [leaf nodes].
      Setting child_branch_factor=2 will be more expensive than the default child_branch_factor=1 (polynomial vs logarithmic), 
        because we traverse 2 children instead of just 1 for each parent node.

  - KeywordTableIndex: 
      by default requires [an LLM] call to extract query keywords.
      Can do index.as_retriever(retriever_mode="simple") 
          or index.as_retriever(retriever_mode="rake") 
      to also use regex/RAKE keyword extractors on your query text.

  - VectorStoreIndex: 
      by default, requires one LLM call per query. 
      If you increase the similarity_top_k or chunk_size, or change the response_mode, then this number will increase.
'''

'''Usage Pattern#
LlamaIndex offers [token predictors] to predict token usage of LLM and embedding calls. 
This allows you to [estimate your costs] during 
  1) index construction, and 
  2) index querying, 
before any respective LLM calls are made.

Tokens are counted using the [TokenCountingHandler] callback. 
See the example notebook for details on the setup.
'''

'''Using MockLLM#
To predict token usage of LLM calls

'''
from llama_index.core.llms import MockLLM
from llama_index.core import Settings

# use a mock llm globally
Settings.llm = MockLLM(max_tokens=256)
# The max_tokens parameter is used as a "worst case" prediction, 
#   where each LLM response will contain exactly that number of tokens. 
# If max_tokens is not specified, then it will simply predict back the prompt

# can use this predictor during
#   index construction 
#   querying.

'''Using MockEmbedding#
You may also predict the token usage of embedding calls with MockEmbedding.
'''
from llama_index.core import MockEmbedding
from llama_index.core import Settings

# use a mock embedding globally
Settings.embed_model = MockEmbedding(embed_dim=1536)


'''Usage Pattern#
'''

'''Estimating LLM and Embedding Token Counts#
need to 
'''
# Setup MockLLM and MockEmbedding objects

from llama_index.core.llms import MockLLM
from llama_index.core import MockEmbedding

llm = MockLLM(max_tokens=256)
embed_model = MockEmbedding(embed_dim=1536)

'''Setup the [TokenCountingCallback] handler'''
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

callback_manager = CallbackManager([token_counter])

'''Add them to the global Settings'''
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
Settings.callback_manager = callback_manager

'''Construct an Index'''
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader(
    "./docs/examples/data/paul_graham"
).load_data()

index = VectorStoreIndex.from_documents(documents)

'''Measure the counts!'''
print(
    "Embedding Tokens: ",
      token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
      token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
      token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
      token_counter.total_llm_token_count,
    "\n",
)

# reset counts
token_counter.reset_counts()

'''Run a query, measure again'''
query_engine = index.as_query_engine()
response = query_engine.query("query")
print(同上面)