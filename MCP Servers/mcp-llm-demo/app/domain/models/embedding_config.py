from pydantic import BaseModel, Field, ConfigDict

class EmbeddingConfig(BaseModel):
    """
    Configuration for text embedding models.
    
    Attributes:
        model_name: Name of the sentence transformer model
        device: Device to run the model on ('cpu', 'cuda', 'auto')
        max_length: Maximum sequence length for embeddings
        normalize_embeddings: Whether to normalize embeddings
        batch_size: Batch size for embedding generation
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    device: str = Field(
        default="auto",
        description="Device to run the model on"
    )
    max_length: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Maximum sequence length"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to normalize embeddings"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Batch size for embedding generation"
    )