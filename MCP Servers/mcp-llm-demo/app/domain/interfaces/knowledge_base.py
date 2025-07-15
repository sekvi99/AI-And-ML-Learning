from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from app.domain.models.document_info import DocumentInfo
from app.domain.models.search_query import SearchQuery
from app.domain.models.search_result import SearchResult


class KnowledgeBase(ABC):
    """
    Abstract base class for knowledge base implementations.
    
    This interface defines the contract for classes that manage
    the complete knowledge base functionality.
    """
    
    @abstractmethod
    async def initialize(self, file_path: Path) -> None:
        """
        Initialize the knowledge base with a document.
        
        Args:
            file_path: Path to the document file
        """
        pass
    
    @abstractmethod
    async def search_knowledge(self, query: SearchQuery) -> List[SearchResult]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query with parameters
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    async def get_info(self) -> DocumentInfo:
        """
        Get information about the knowledge base.
        
        Returns:
            Document information
        """
        pass
    
    @abstractmethod
    async def reload(self) -> None:
        """Reload the knowledge base from the source document."""
        pass