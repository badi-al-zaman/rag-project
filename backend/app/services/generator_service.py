from llama_index.core.agent import FunctionAgent
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from typing import List, Dict
# from app.models import Message
from llama_index.core.schema import TextNode
from app.utils.logger import logger

from app.services.retriever_service import index as wiki_index_v2

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


async def llm_chat_v2(prompt: str, chat_memory):
    query_engine = wiki_index_v2.as_chat_engine(chat_mode=ChatMode.BEST, memory=chat_memory)
    response = await query_engine.achat(prompt)  # chat_history=messages
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


async def search_documents(query: str) -> str:
    """Search through wiki articles about John Adams and James Monroe."""
    if query.strip() != "":
        logger.info(f"search using {query}")
        query_engine = wiki_index_v2.as_query_engine()
        response = await query_engine.aquery(query)
        return str(response)
    return ""


from llama_index.core.agent.workflow import AgentWorkflow

#
# ### Response Guidelines
# 1. **Clarity & Accuracy:** Provide concise, factual, and verifiable answers based on the search results.
# 2. **Citation:** When possible, mention the source context (e.g., “According to the John Adams article…”).
# 3. **Reasoning:** If search results are unclear or conflicting, summarize what is known and note uncertainties.
# 4. **Relevance:** Only include information related to the query topic. Exclude unrelated content.
# 5. **Style:** Write in a neutral, informative tone — similar to a historical researcher explaining a finding.


system_prompt = """
You are an intelligent research assistant specializing in topics related to John Adams and James Monroe.

Your main goal is to accurately answer user questions using information retrieved from a document search tool.

### Capabilities
- You can call the `search_documents(query)` function to look up information from a curated collection of wiki articles about John Adams and James Monroe.

important: Before answering, decide if document search tool is needed"
    - Use `search_documents` only if the question involves factual, historical, or specific details about John Adams or James Monroe.
    - If the question is general (e.g., “Who were they?” or “What is their legacy?”), use your internal knowledge.
    - synthesize search results clearly and avoid repeating irrelevant text from the source.

### Output Format
- Provide a direct answer first.
- Optionally, follow with a short “Explanation” or “Summary of findings” section.
"""
system_prompt_2 = """You are a helpful assistant that can perform search through documents to answer questions."""

# Create an enhanced workflow with tools
# agent = AgentWorkflow.from_tools_or_functions(
#     [search_documents],
#     llm=model,
#     system_prompt=system_prompt,
# )

# https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools#return-direct
# tool = QueryEngineTool.from_defaults(
#     query_engine,
#     name="<name>",
#     description="<description>",
#     return_direct=True,
# )

workflow = FunctionAgent(
    tools=[search_documents],
    llm=model,
    system_prompt=system_prompt,
)


# Now we can ask questions about the documents
async def ask_agent(query: str, chat_memory):
    response = await workflow.run(
        query, memory=chat_memory
    )
    return response
