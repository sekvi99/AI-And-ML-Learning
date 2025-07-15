from typing import Any, Dict, Optional
from app.domain.models.processing_status import ProcessingStatus
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path


class DocumentInfo(BaseModel):
    """
    Information about a processed document.
    
    Attributes:
        path: Path to the document file
        exists: Whether the file exists
        total_chunks: Total number of chunks created
        chunk_size: Size of each chunk in characters
        status: Current processing status
        error_message: Error message if processing failed
        metadata: Additional document metadata
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    path: Path = Field(..., description="Path to the document file")
    exists: bool = Field(..., description="Whether the file exists")
    total_chunks: int = Field(
        default=0,
        ge=0,
        description="Total number of chunks created"
    )
    chunk_size: int = Field(
        default=500,
        ge=1,
        description="Size of each chunk in characters"
    )
    status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current processing status"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document metadata"
    )