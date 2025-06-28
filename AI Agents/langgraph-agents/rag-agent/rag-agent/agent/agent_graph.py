from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from operator import add as add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

def build_agent_graph(llm, tools, system_prompt):
    tools_dict = {tool.name: tool for tool in tools}

    def call_llm(state: AgentState) -> AgentState:
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages
        message = llm.invoke(messages)
        return {'messages': [message]}

    def take_action(state: AgentState) -> AgentState:
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if t['name'] not in tools_dict:
                result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    return graph.compile()