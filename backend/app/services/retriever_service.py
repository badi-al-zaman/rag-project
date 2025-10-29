import chromadb
from chromadb.config import Settings


# ========================================
# SECTION 4: VECTOR SEARCH
# ========================================

def search_vector_database(collection, query_embedding, top_k: int = 3):
    """
    Search vector database for relevant document chunks.

    This section demonstrates:
    - Vector similarity search
    - Result ranking and filtering
    - Similarity scoring
    - Top-k result selection
    """
    # Perform vector search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,  # How many results are returned?
    )
    # Process and display results
    search_results = []
    for i, (doc_id, distance, content, metadata) in enumerate(
            zip(
                results["ids"][0],
                results["distances"][0],
                results["documents"][0],
                results["metadatas"][0],
            )
    ):
        similarity = 1 - distance  # Convert distance to similarity
        search_results.append(
            {
                "id": doc_id,
                "content": content,
                "title": content,
                "metadata": metadata,
                "similarity": similarity,
            }
        )

    return search_results


# ========================================
# Version 2 setup using llama-index framework
# ========================================

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient("vector_db/chroma", settings=Settings(anonymized_telemetry=False))

# Create collection (what is the collection name? | What similarity metric is used? | embedding_functions)
chroma_collection = chroma_client.get_or_create_collection(name="wiki_articles_v2", metadata={
    "hnsw:space": "cosine"})

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# load embedded chunks from chroma -collection
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)


def search_query_pipline_v2(query: str, custom_index=index):
    retriever = custom_index.as_retriever(similarity_top_k=3)
    # Step 4: Search vector database
    response = retriever.retrieve(query)
    return response
