from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from typing import List, Dict
# from app.models import Message
from llama_index.core.schema import TextNode

from app.services.retriever_service import index

model = Ollama(
    model="llama3.1:8b",  # local model name
    request_timeout=360.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)


# ========================================
# SECTION 5: CONTEXT AUGMENTATION
# ========================================

def augment_prompt_with_context(query: str, search_results: List[Dict]) -> str:
    """
    Build augmented prompt with retrieved context for LLM.

    This section demonstrates:
    - Context assembly from search results
    - Prompt construction
    - Information formatting
    - Context length management
    """
    # Assemble context from search results
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(
            f"Source {i}: {result['metadata']['title']}\n{result['content']}"
        )

    context = "\n\n".join(context_parts)

    # Build augmented prompt
    augmented_prompt = f"""
Based on the following resources, answer the user's question.

resources:
{context}

QUESTION: {query}

Please provide a clear, accurate answer based on the resources above.
If the information is not available in the resources, say so.
Include relevant resources details and any limitations or requirements.
"""
    return augmented_prompt


# Now we can ask questions about the documents
def generator_complete(augmented_prompt: str):
    response = model.complete(augmented_prompt)
    return str(response)


def llm_chat(prompt: str, session):
    # messages = [
    #     ChatMessage(
    #         role="system", content="You are a pirate with a colorful personality"
    #     ),
    #     ChatMessage(role="user", content="What is your name"),
    # ]
    messages = []
    for message in session["messages"]:
        if len(message["retrieved_docs"]) != 0:
            messages.append(ChatMessage(
                role=message["role"], content=augment_prompt_with_context(prompt, message["retrieved_docs"])
            ))
        else:
            messages.append(ChatMessage(
                role=message["role"], content=message["content"]
            ))
    response = model.chat(messages=messages)
    return str(response)


def llm_chat_v2(prompt: str, chat_memory):
    query_engine = index.as_chat_engine(chat_mode=ChatMode.BEST, memory=chat_memory)
    response = query_engine.chat(prompt)  # chat_history=messages
    return response


# ========================================
# SECTION 6: RESPONSE GENERATION
# ========================================


def generate_response(augmented_prompt: str) -> str:
    """
    Generate response using LLM (simulated for demo).

    This section demonstrates:
    - LLM integration (simulated)
    - Response formatting
    - Answer synthesis
    - Output structure
    """
    # LLM processing...

    response = model.chat(messages=[ChatMessage(
        role="user", content=augmented_prompt)
    ])
    return str(response)
    # response = generator_complete(augmented_prompt)

    return response
