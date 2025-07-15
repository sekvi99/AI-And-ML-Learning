from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict

class SearchResult(BaseModel):
    """
    Represents a single search result from the knowledge base.
    
    Attributes:
        text: The content of the text chunk
        similarity: Cosine similarity score between query and chunk (0.0 to 1.0)
        chunk_id: Unique identifier for the chunk within the document
        metadata: Additional metadata about the chunk (page number, etc.)
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    text: str = Field(..., description="The text content of the search result")
    similarity: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Similarity score between 0.0 and 1.0"
    )
    chunk_id: int = Field(..., ge=0, description="Unique chunk identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the chunk"
    )