import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

# Define the state structure for the agent
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]  # Conversation history

# Initialize the OpenAI language model (GPT-4o)
llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    """
    This node will solve the request you input.
    It takes the current state, sends the conversation to the LLM,
    appends the AI's response to the state, and returns the updated state.
    """
    response = llm.invoke(state["messages"])  # Get AI response
    state["messages"].append(AIMessage(content=response.content))  # Add AI response to history
    print(f"AI: {response.content}")  # Print AI response
    print("CURRENT STATE:", state["messages"])  # Print current conversation state
    return state  # Return updated state

# Create a stateful graph for the agent's workflow
graph = StateGraph(AgentState)
graph.add_node("process", process)  # Add processing node
graph.add_edge(START, "process")    # Start -> process
graph.add_edge("process", END)      # process -> End
agent = graph.compile()             # Compile the agent graph

def save_conversation_history(history: List[Union[HumanMessage, AIMessage]], filename: str = "conversation_history.txt") -> None:
    """
    Save the conversation history to a text file.
    """
    with open(filename, "w") as file:
        for message in history:
            if isinstance(message, HumanMessage):
                file.write(f"You: {message.content}\n")
            elif isinstance(message, AIMessage):
                file.write(f"AI: {message.content}\n")

    print(f"Conversation history saved to {filename}")

def read_conversation_history(filename: str = "conversation_history.txt") -> List[Union[HumanMessage, AIMessage]]:
    """
    Read the conversation history from a text file.
    """
    history = []
    if os.path.exists(filename):
        with open(filename, "r") as file:
            for line in file:
                if line.startswith("You:"):
                    history.append(HumanMessage(content=line.strip().split("You: ")[1]))
                elif line.startswith("AI:"):
                    history.append(AIMessage(content=line.strip().split("AI: ")[1]))
    return history

def main() -> None:
    history = read_conversation_history()  # Load existing conversation history
    print(history)
    if history:
        converstaion_history = history  # Store conversation history
    else:
        converstaion_history = []
    print("Hello from memory-agent!")
    
    user_input = input("You: ")  # Get initial user input
    while user_input.lower() != "exit":  # Loop until user types 'exit'
        converstaion_history.append(HumanMessage(content=user_input))  # Add user message to history
        result = agent.invoke({"messages": converstaion_history})      # Run agent with current history
        print(result["messages"])                                      # Print updated messages
        converstaion_history = result["messages"]                      # Update history with AI response
        
        user_input = input("You: ")  # Get next user input
    
    save_conversation_history(converstaion_history)  # Save conversation history to file

if __name__ == "__main__":
    main()  # Run the main function if script is
