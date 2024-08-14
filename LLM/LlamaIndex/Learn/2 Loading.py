'''
Loading Data (Ingestion)

  ingestion pipeline typically consists of three main stages:

    Load the data
    Transform the data
    Index and store the data (此篇先不提)


  Loaders#
    Before your chosen LLM can act on your data you need to load it. 
    data connectors, also called Reader. 
    Data connectors ingest data from different data sources and format the data into Document objects. 
    A Document is a collection of data ( text, images , audio ...) and metadata about that data.


'''
    # SimpleDirectoryReader, 
    # creates documents out of every file in a given directory. 
    # can read a variety of formats including 
    #   Markdown, PDFs, Word documents, PowerPoint decks, images, audio and video.

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()


'''
  Using Readers from LlamaHub
    not all built-in. Instead, you download them from our registry of data connectors, LlamaHub.

    example :
    LlamaIndex downloads and installs the connector called DatabaseReader, 
    which runs a query against a SQL database and returns every row of the results as a Document:
I'''

from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    scheme=os.getenv("DB_SCHEME"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    dbname=os.getenv("DB_NAME"),
)

query = "SELECT * FROM users"
documents = reader.load_data(query=query)


# Create a Document directly.
from llama_index.core import Document
doc = Document(text="text")


'''
Transformations#
  After the data is loaded, 
  process and transform your data before putting it into a storage system. 
  These transformations include 
    chunking, 
    extracting metadata, 
    and embedding each chunk. 
  This is necessary to make sure that the data can be retrieved, and used optimally by the LLM.

  Transformation input/outputs are Node objects (a Document is a subclass of a Node). Transformations can also be stacked and reordered.
'''


'''
  High-Level Transformation API
    Indexes have a .from_documents() method
      accepts an array of Document objects and will correctly parse and chunk them up. 
'''
vector_index = VectorStoreIndex.from_documents(documents)
vector_index.as_query_engine()
# Under the hood, this splits your Document into Node objects

# customize :
#  like the text splitter, through this abstraction you can pass in a 
#   custom transformations list or apply to the global Settings:
from llama_index.core.node_parser import SentenceSplitter
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

# global
from llama_index.core import Settings
Settings.text_splitter = text_splitter

# per-index
index = VectorStoreIndex.from_documents(
    documents, transformations=[text_splitter]
)

'''Lower-Level Transformation API
either using our transformation modules (text splitters, metadata extractors, etc.) as standalone components, 
or compose them in our declarative Transformation Pipeline interface.
'''
'''Splitting Your Documents into Nodes#
  bite-sized pieces that can be retrieved / fed to the LLM.

  text splitters, 
  ranging from paragraph/sentence/token based splitters 
  to file-based splitters like HTML, JSON.
'''
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter

documents = SimpleDirectoryReader("./data").load_data()

pipeline = IngestionPipeline(transformations=[TokenTextSplitter(), ...])

nodes = pipeline.run(documents=documents)


'''Adding Metadata
can add metadata to your documents and nodes
guides on 1) how to customize Documents, and 2) how to customize Nodes.
'''
document = Document(
    text="text",
    metadata={"filename": "<doc_file_name>", "category": "<category>"},
)

'''Adding Embeddings#
See our ingestion pipeline or our embeddings guide for more details.
'''

'''
Creating and passing Nodes directly#
'''
from llama_index.core.schema import TextNode
node1 = TextNode(text="<text_chunk>", id_="<node_id>")
node2 = TextNode(text="<text_chunk>", id_="<node_id>")

index = VectorStoreIndex([node1, node2])



# Built-inSimpleDirectoryReader. Can support parsing a wide range of file types including .md, .pdf, .jpg, .png, .docx, as well as audio and video types.
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data").load_data()

# LlamaHub contains a registry of open-source data connectors 
# Usage Pattern
from llama_index.core import download_loader

from llama_index.readers.google import GoogleDocsReader

loader = GoogleDocsReader()
documents = loader.load_data(document_ids=[...])

