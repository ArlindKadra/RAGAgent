import os
from typing import TypedDict, Annotated

import datasets
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.tools import Tool

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline

# Windows only since it has a triton error
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"


def main():

    # Load the dataset
    guest_dataset = datasets.load_dataset(
        "agents-course/unit3-invitees",
        split="train",
    )

    # Convert dataset entries into Document objects
    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]

    bm25_retriever = BM25Retriever.from_documents(docs)

    def extract_text(query: str) -> str:
        """Retrieves detailed information about gala guests based on their name or relation."""
        results = bm25_retriever.invoke(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found."

    guest_info_tool = Tool(
        name="guest_info_retriever",
        func=extract_text,
        description="Retrieves detailed information about gala guests based on their name or relation."
    )

    """
    # Generate the chat interface, including the tools
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    """
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/gemma-3-270m-it",
        task="text-generation",
        model_kwargs={
            "temperature":0.1
        },
    )

    chat = ChatHuggingFace(llm=llm, verbose=True)
    tools = [guest_info_tool]
    chat_with_tools = chat.bind_tools(tools)

    # Generate the AgentState and Agent graph
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def assistant(state: AgentState):
        return {
            "messages": [chat_with_tools.invoke(state["messages"])],
        }

    ## The graph
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message requires a tool, route to tools
        # Otherwise, provide a direct response
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    jarvis = builder.compile()

    messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
    response = jarvis.invoke({"messages": messages})

    print("🎩 Jarvis's Response:")
    print(response['messages'][-1].content)


if __name__ == "__main__":
    main()
