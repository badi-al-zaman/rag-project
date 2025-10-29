# ========================================
# SECTION 7: COMPLETE RAG PIPELINE
# ========================================
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.services.embedding_service import load_and_chunk_documents, load_and_chunk_documents_v2
from app.services.embedding_service import setup_vector_database, process_user_query, setup_vector_database_v2
from app.services.retriever_service import search_vector_database, search_query_pipline_v2, index
from app.services.generator_service import augment_prompt_with_context, generate_response
from llama_index.llms.ollama import Ollama

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.llm = Ollama(
    model="llama3.1:8b",  # local model name
    request_timeout=360.0,
    context_window=8000
)


def run_complete_rag_pipeline(query: str):
    """
    Run the complete RAG pipeline from start to finish.

    This demonstrates the full flow:
    1. Document loading and chunking
    2. Vector database setup
    3. Query processing
    4. Vector search
    5. Context augmentation
    6. Response generation
    """
    # Step 1: Load and chunk documents
    chunks = load_and_chunk_documents()

    # Step 2: Setup vector database
    collection = setup_vector_database(chunks)

    # Step 3: Process user query
    model, query_embedding = process_user_query(query)

    # Step 4: Search vector database
    search_results = search_vector_database(collection, query_embedding, top_k=3)

    # Step 5: Augment prompt with context
    augmented_prompt = augment_prompt_with_context(query, search_results)

    # Step 6: Generate response
    response = generate_response(augmented_prompt)

    return response


def run_complete_rag_pipeline_v2(query: str):
    """
    Run the complete RAG pipeline from start to finish.
    This demonstrates the full flow:
    1. Document loading and chunking
    2. Vector database setup
    3. Query processing
    4. Vector search
    5. Context augmentation
    6. Response generation
    """
    retriever = index.as_query_engine(similarity_top_k=3)
    return retriever.query(query)

    # # Step 3: Process user query
    # # Step 4: Search vector database
    # search_results = search_query_pipline_v2(query, index)

    # # Step 5: Augment prompt with context
    # augmented_prompt = augment_prompt_with_context(query, search_results)
    #
    # # Step 6: Generate response
    # response = generate_response(augmented_prompt)
