from typing import Any, Dict
from pydantic import BaseModel, Field, ConfigDict

class ToolCallRequest(BaseModel):
    """
    Represents a tool call request.
    
    Attributes:
        name: Name of the tool to call
        arguments: Arguments for the tool call
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments"
    )