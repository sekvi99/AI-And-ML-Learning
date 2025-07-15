from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict

class TextChunk(BaseModel):
    """
    Represents a chunk of text from a document.
    
    Attributes:
        id: Unique identifier for the chunk
        text: The text content of the chunk
        page_number: Page number where the chunk originates (if applicable)
        start_position: Starting character position in the original document
        end_position: Ending character position in the original document
        metadata: Additional metadata about the chunk
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: int = Field(..., ge=0, description="Unique chunk identifier")
    text: str = Field(..., min_length=1, description="Text content of the chunk")
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Page number where chunk originates"
    )
    start_position: Optional[int] = Field(
        default=None,
        ge=0,
        description="Starting character position"
    )
    end_position: Optional[int] = Field(
        default=None,
        ge=0,
        description="Ending character position"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk metadata"
    )