from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class ToolCallResponse(BaseModel):
    """
    Represents a tool call response.
    
    Attributes:
        success: Whether the tool call was successful
        content: Response content
        error: Error message if the call failed
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    success: bool = Field(..., description="Whether the call was successful")
    content: str = Field(..., description="Response content")
    error: Optional[str] = Field(
        default=None,
        description="Error message if call failed"
    )