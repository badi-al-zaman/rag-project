# ========================================
# SECTION 7: COMPLETE RAG PIPELINE
# ========================================


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
    print("\n🚀 COMPLETE RAG PIPELINE DEMO")
    print("=" * 60)
    print(f"❓ User Question: {query}")
    print("=" * 60)

    # Step 1: Load and chunk documents
    chunks = load_and_chunk_documents()

    # Step 2: Setup vector database
    collection = setup_vector_database(chunks)

    # Step 3: Process user query
    model, query_embedding = process_user_query(query)

    # Step 4: Search vector database
    search_results = search_vector_database(collection, query_embedding)

    # Step 5: Augment prompt with context
    augmented_prompt = augment_prompt_with_context(query, search_results)

    # Step 6: Generate response
    response = generate_response(augmented_prompt)

    # Display final result
    print("\n🎉 FINAL RESULT")
    print("=" * 60)
    print(response)
    print("=" * 60)

    return response