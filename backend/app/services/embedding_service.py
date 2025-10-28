from typing import List, Dict
from chromadb.config import Settings
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document


# ========================================
# SECTION 1: DOCUMENT LOADING & CHUNKING
# ========================================

def load_documents(path="./app/data/articles"):
    # Load documents from directory
    documents = SimpleDirectoryReader(
        input_dir=path,
    ).load_data()

    wiki_articles = []
    for document in documents:
        current_document = document.dict()
        # Sample article documents
        wiki_articles.append({
            "id": current_document["id_"],
            "title": current_document['metadata']['file_name'],
            "content": current_document['text'],
        })

    return wiki_articles


def load_and_chunk_documents(path="./app/data/articles"):
    """
    Load sample documents and chunk them for better retrieval.
    This section demonstrates:
    - Document loading from sample data
    - Text chunking using LangChain
    - Chunk size and overlap configuration
    """
    wiki_articles = load_documents(path)

    # Configure text splitter from langchain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # What is the chunk size?
        chunk_overlap=50,  # What is the overlap?
        length_function=len,
        # separators=["\n\n", "\n", " ", ""], # default values
    )

    # Chunk all documents
    all_chunks = []
    for doc in wiki_articles:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{doc['id']}_chunk_{i}",
                    "title": doc["title"],
                    "content": chunk,
                    "source_doc": doc["id"],
                }
            )
    return all_chunks


# ========================================
# SECTION 2: VECTOR DATABASE SETUP
# ========================================

def setup_vector_database(chunks: List[Dict]):
    """
    Set up ChromaDB vector database and store document chunks.

    This section demonstrates:
    - ChromaDB client initialization
    - Collection creation
    - Document embedding and storage
    - Vector database configuration
    """
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient("vector_db/chroma", settings=Settings(anonymized_telemetry=False))

    # Create collection (what is the collection name? | What similarity metric is used? | embedding_functions)
    collection = chroma_client.get_or_create_collection(name="wiki_articles_v1", metadata={
        "hnsw:space": "cosine"}, )

    # Add documents to collection (embeddings will be generated automatically)
    if collection.count() == 0:
        # Prepare data for storage
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "title": chunk["title"],
                "source": chunk["source_doc"],
            }
            for chunk in chunks
        ]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    else:
        print(f"✅ Collection already contains {collection.count()} chunks")
    return collection


# ========================================
# SECTION 3: QUERY PROCESSING
# ========================================
from sentence_transformers import SentenceTransformer


def process_user_query(query: str):
    """
    Process user query and convert to embedding for vector search.

    This section demonstrates:
    - Query preprocessing
    - Embedding model usage
    - Vector conversion
    - Query optimization
    """
    # Load embedding model (what model is used?)
    model = SentenceTransformer("all-MiniLM-L6-v2")  # What embedding model is used? same as chroma vector store

    # Preprocess query
    cleaned_query = query.lower().strip()

    # Convert query to embedding
    query_embedding = model.encode([cleaned_query])

    return model, query_embedding[0]


# ========================================
# Version 2 setup using llama-index framework
# ========================================
from llama_index.core.extractors import TitleExtractor
from llama_index.llms.ollama import Ollama

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Node


class Title(BaseModel):
    """A Document Title """
    title: str


def load_and_chunk_documents_v2(vector_store, path="./app/data/articles"):
    """
    Load sample documents and apply text transformations then chunks them for better retrieval.
    This section demonstrates:
    - Document loading from sample data
    - Text chunking using pipe ingestion of llama-index
    - Chunk size and overlap configuration
    - generate chunks embeddings & store them
    """

    # Load documents from directory
    documents = SimpleDirectoryReader(
        input_dir=path,
    ).load_data()

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # configure LLM model to use it for extracting documents' metadata like: title,....
    # llm = Ollama(
    #     model="llama3.1:8b",  # local model name
    #     request_timeout=360.0,
    #     context_window=8000
    # )
    # sllm = llm.as_structured_llm(Title)

    # create the chunks' pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=200, chunk_overlap=50),
            # TitleExtractor(llm=sllm),
            embed_model,
        ],
        vector_store=vector_store,
        docstore=SimpleDocumentStore()
    )

    # run the pipeline to generate documents embeddings and store them
    pipeline.run(documents=documents, num_workers=4)

    # print(nodes[0].extra_info["document_title"])

    # save -- Local Cache Management --
    pipeline.persist("cache/pipeline_storage")

    # index = VectorStoreIndex.from_vector_store(vector_store)
    # return index


# def get_index():
#     # Step 1: Setup vector database
#     vector_store = setup_vector_database_v2()
#
#     # storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     index = VectorStoreIndex.from_vector_store(
#         vector_store, embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
#     )
#     return index

def setup_vector_database_v2(chunks: List[Node] | None = None):
    """
    Set up ChromaDB vector database and store document chunks.

    This section demonstrates:
    - ChromaDB client initialization
    - Collection creation
    - Vector database configuration
    """
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient("vector_db/chroma", settings=Settings(anonymized_telemetry=False))

    # Create collection (what is the collection name? | What similarity metric is used? | embedding_functions)
    chroma_collection = chroma_client.get_or_create_collection(name="wiki_articles_v2", metadata={
        "hnsw:space": "cosine"})

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #
    # index = VectorStoreIndex.build_index_from_nodes(
    #     nodes=chunks, storage_context=storage_context, embed_model=embed_model
    # )

    return vector_store
