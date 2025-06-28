"""
agent_state.py

Defines the AgentState TypedDict for message passing.
"""

from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(dict):
    """
    AgentState holds the sequence of messages exchanged in the agent's workflow.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]