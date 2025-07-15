from abc import ABC, abstractmethod
from typing import List

from app.domain.models.search_query import SearchQuery
from app.domain.models.search_result import SearchResult
from app.domain.models.text_chunk import TextChunk

import numpy as np

class SearchEngine(ABC):
    """
    Abstract base class for search engine implementations.
    
    This interface defines the contract for classes that can perform
    semantic search over text embeddings.
    """
    
    @abstractmethod
    async def index_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """
        Index text chunks with their embeddings for search.
        
        Args:
            chunks: List of text chunks to index
            embeddings: Corresponding embeddings for the chunks
        """
        pass
    
    @abstractmethod
    async def search(self, query: SearchQuery, query_embedding: np.ndarray) -> List[SearchResult]:
        """
        Perform semantic search using the query embedding.
        
        Args:
            query: Search query with parameters
            query_embedding: Embedding representation of the query
            
        Returns:
            List of search results ranked by similarity
        """
        pass
    
    @abstractmethod
    async def clear_index(self) -> None:
        """Clear the search index."""
        pass
    
    @abstractmethod
    def get_index_size(self) -> int:
        """
        Get the number of items in the search index.
        
        Returns:
            Number of indexed items
        """
        pass