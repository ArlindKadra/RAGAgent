import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

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
    checkpointer = MemorySaver()
    jarvis = builder.compile(checkpointer=checkpointer)

    messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
    response = jarvis.invoke({"messages": messages},  config={"configurable": {"thread_id": "admin1"}})

    print("🎩 Jarvis's Response:")
    print(response['messages'][-1].content)

    messages = [HumanMessage(content="Could you generate a few topics based on her interests as ice-breakers?.")]
    response = jarvis.invoke({"messages": messages},  config={"configurable": {"thread_id": "admin1"}})

    print("🎩 Jarvis's Response:")
    print(response['messages'][-1].content)


if __name__ == "__main__":
    main()
