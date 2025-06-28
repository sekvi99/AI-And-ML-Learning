"""
drafter_agent.py

Contains the main agent logic, graph setup, and utility functions.
"""

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent_tools import update, save, DOCUMENT_CONTENT
from agent_state import AgentState

# List of available tools for the agent
tools = [update, save]

# Bind tools to the model
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    """
    Main agent function that handles user input and invokes the model.

    Args:
        state (AgentState): Current state containing message history.

    Returns:
        AgentState: Updated state with new messages.
    """
    # System prompt to instruct the agent
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.

    The current document content is:{DOCUMENT_CONTENT}
    """)

    # If no messages yet, start the conversation
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """
    Determines whether to continue or end the conversation.

    Args:
        state (AgentState): Current state.

    Returns:
        str: "continue" or "end"
    """
    messages = state["messages"]
    if not messages:
        return "continue"
    # Check if the last tool message indicates the document was saved
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
    return "continue"

def print_messages(messages):
    """
    Prints the last few messages in a readable format.

    Args:
        messages (list): List of messages.
    """
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

def build_graph():
    """
    Builds and compiles the agent's workflow graph.

    Returns:
        app: Compiled graph application.
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", our_agent)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_edge("agent", "tools")
    graph.add_conditional_edges(
        "tools",
        should_continue,
        {
            "continue": "agent",
            "end": END,
        },
    )
    return graph.compile()

def run_document_agent():
    """
    Runs the Drafter agent in a loop until the document is saved.
    """
    print("\n ===== DRAFTER =====")
    app = build_graph()
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n ===== DRAFTER FINISHED =====")