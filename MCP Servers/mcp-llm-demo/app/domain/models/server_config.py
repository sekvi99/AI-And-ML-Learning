from app.domain.models.chunking_config import ChunkingConfig
from app.domain.models.embedding_config import EmbeddingConfig
from pydantic import BaseModel, Field, ConfigDict

class ServerConfig(BaseModel):
    """
    Configuration for the MCP server.
    
    Attributes:
        name: Server name
        version: Server version
        description: Server description
        embedding_config: Configuration for embeddings
        chunking_config: Configuration for text chunking
        debug: Enable debug mode
        log_level: Logging level
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    name: str = Field(default="pdf-knowledge-server", description="Server name")
    version: str = Field(default="1.0.0", description="Server version")
    description: str = Field(
        default="PDF Knowledge Base Server with semantic search",
        description="Server description"
    )
    embedding_config: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding configuration"
    )
    chunking_config: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
        description="Chunking configuration"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )