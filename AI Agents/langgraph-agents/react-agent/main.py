# Import necessary modules and classes for typing, environment variables, LangChain, and LangGraph
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # Base class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Used to pass tool call results back to the LLM
from langchain_core.messages import SystemMessage # Used to provide instructions to the LLM
from langchain_openai import ChatOpenAI # OpenAI chat model wrapper
from langchain_core.tools import tool # Decorator to define tools
from langgraph.graph.message import add_messages # Utility for message state management
from langgraph.graph import StateGraph, END # StateGraph for workflow, END marks the end node
from langgraph.prebuilt import ToolNode # Prebuilt node for tool execution

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Define the agent's state, which consists of a sequence of messages
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define three simple math tools using the @tool decorator

@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""
    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

# List of available tools for the agent
tools = [add, subtract, multiply]

# Initialize the OpenAI chat model and bind the tools to it
model = ChatOpenAI(model = "gpt-4o").bind_tools(tools)

# Function to call the model with the current state and a system prompt
def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Function to determine whether the agent should continue (call a tool) or end
def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    # If the last message has tool calls, continue; otherwise, end
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
    
# Build the agent workflow graph
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call) # Node for the agent's reasoning step

tool_node = ToolNode(tools=tools) # Node for tool execution
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent") # Set the starting node

# Add conditional edges: if the agent wants to call a tool, go to the tool node; else, end
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

# After tool execution, return to the agent node
graph.add_edge("tools", "our_agent")

# Compile the graph into an executable app
app = graph.compile()

# Helper function to print the output stream from the agent
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# Example input: ask the agent to perform math and tell a joke
inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))