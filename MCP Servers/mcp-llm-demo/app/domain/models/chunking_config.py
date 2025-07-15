from typing import Any, Dict
from pydantic import BaseModel, Field, ConfigDict, validator

class ChunkingConfig(BaseModel):
    """
    Configuration for text chunking strategy.
    
    Attributes:
        chunk_size: Target size for each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for a chunk to be valid
        split_strategy: Strategy for splitting text ('sentence', 'paragraph', 'fixed')
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    chunk_size: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Target chunk size in characters"
    )
    overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Character overlap between chunks"
    )
    min_chunk_size: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Minimum chunk size in characters"
    )
    split_strategy: str = Field(
        default="sentence",
        description="Text splitting strategy"
    )
    
    @validator('overlap')
    def validate_overlap(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure overlap is less than chunk_size."""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("Overlap must be less than chunk_size")
        return v