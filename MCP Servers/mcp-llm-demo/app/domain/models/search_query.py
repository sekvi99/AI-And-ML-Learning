from pydantic import BaseModel, Field, ConfigDict

class SearchQuery(BaseModel):
    """
    Represents a search query with parameters.
    
    Attributes:
        query: The search query string
        top_k: Maximum number of results to return
        min_similarity: Minimum similarity threshold for results
        include_metadata: Whether to include metadata in results
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(
        default=5, 
        ge=1, 
        le=50, 
        description="Number of top results to return"
    )
    min_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in search results"
    )