# is key to understanding and optimizing it. 

'''Basic logging#
turn on debug logging. 
can be done anywhere in your application like this:
'''
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


'''Callback handler#
Using the callback manager, as many callbacks as needed can be added.

can also track the [duration] and [number of occurrences] [of each event].

a trace map of events is also recorded, and callbacks can use this data

For example, the LlamaDebugHandler will, by default, print the trace of events after most operations.

'''
# You can get a simple callback handler like this:
import llama_index.core

llama_index.core.set_global_handler("simple")
# learn how to build you own custom callback handler.

'''Observability#
LlamaIndex provides [one-click observability] to allow you to build principled LLM applications in a production setting.

This feature allows you to seamlessly integrate the LlamaIndex library with powerful observability/evaluation tools offered by our partners. 
Configure a variable once, and you'll be able to do things like the following:

  - View LLM/prompt inputs/outputs
  - Ensure that the outputs of any component (LLMs, embeddings) are performing as expected
  - View call traces for both indexing and querying

To learn more, check out our observability docs
'''