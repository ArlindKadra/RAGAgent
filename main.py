import os
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from tools import extract_text

# Windows only since it has a triton error
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"


def main():

    llm = HuggingFacePipeline.from_model_id(
        model_id="google/gemma-3-270m-it",
        task="text-generation",
        model_kwargs={
            "temperature":0.1
        },
    )

    chat = ChatHuggingFace(llm=llm, verbose=True)
    tools = [extract_text]
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
