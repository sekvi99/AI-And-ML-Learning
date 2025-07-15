from abc import ABC, abstractmethod
from typing import List
import numpy as np

from app.domain.models.embedding_config import EmbeddingConfig


class EmbeddingProvider(ABC):
    """
    Abstract base class for text embedding implementations.
    
    This interface defines the contract for classes that can generate
    vector embeddings from text for semantic search.
    """
    
    @abstractmethod
    async def initialize(self, config: EmbeddingConfig) -> None:
        """
        Initialize the embedding provider with the given configuration.
        
        Args:
            config: Configuration for the embedding model
        """
        pass
    
    @abstractmethod
    async def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings
        """
        pass
    
    @abstractmethod
    async def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query string
            
        Returns:
            Numpy array representing the query embedding
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this provider.
        
        Returns:
            Embedding dimension
        """
        pass