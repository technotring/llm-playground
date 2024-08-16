from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="llama3.1", request_timeout=360.0)

Settings.chunk_size = 512

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=512)])
query_engine = index.as_query_engine(similarity_top_k=5, response_mode="tree_summarize", streaming=True)
response = query_engine.query("What did the author do growing up?")
print(response)
