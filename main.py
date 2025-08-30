import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from yaml import safe_load

from tools import extract_text, search_tool

# Windows only since it has a triton error
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

PROMPT_TEMPLATE_FILE = "prompts.yaml"


def main():

    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        huggingfacehub_api_token=os.environ["HUGGINGFACE_API_KEY"],
        provider='auto'
    )

    with open(PROMPT_TEMPLATE_FILE, 'r') as stream:
        prompt_data = safe_load(stream)

    system_message = prompt_data["system_prompt"]
    chat = ChatHuggingFace(llm=llm, verbose=True)
    tools = [extract_text, search_tool]
    chat_with_tools = chat.bind_tools(tools)

    system_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("messages"),  # inject state messages
    ])

    system_chain = system_prompt | chat_with_tools

    # Generate the AgentState and Agent graph
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def assistant(state: AgentState):
        return {
            "messages": [system_chain.invoke(state["messages"])],
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

    messages = [HumanMessage(content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?")]
    response = jarvis.invoke({"messages": messages},  config={"configurable": {"thread_id": "admin1"}})

    print("🎩 Jarvis's Response:")

    print(response['messages'][-1].content)

    messages = [HumanMessage(content="Could you generate a few topics as ice-breakers based on our conversation so far?.")]
    response = jarvis.invoke({"messages": messages},  config={"configurable": {"thread_id": "admin1"}})

    print("🎩 Jarvis's Response:")
    print(response['messages'][-1].content)


if __name__ == "__main__":
    main()
