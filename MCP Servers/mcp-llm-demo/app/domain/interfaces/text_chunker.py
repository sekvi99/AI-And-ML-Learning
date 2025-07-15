from abc import ABC, abstractmethod
from typing import Any, Dict, List

from app.domain.models.chunking_config import ChunkingConfig
from app.domain.models.text_chunk import TextChunk


class TextChunker(ABC):
    """
    Abstract base class for text chunking implementations.
    
    This interface defines the contract for classes that can split text
    into smaller, manageable chunks for processing.
    """
    
    @abstractmethod
    def chunk_text(self, text: str, config: ChunkingConfig) -> List[TextChunk]:
        """
        Split text into chunks based on the provided configuration.
        
        Args:
            text: The text to be chunked
            config: Configuration for chunking strategy
            
        Returns:
            List of text chunks
        """
        pass
    
    @abstractmethod
    def get_chunk_metadata(self, chunk: TextChunk, original_text: str) -> Dict[str, Any]:
        """
        Extract metadata for a text chunk.
        
        Args:
            chunk: The text chunk
            original_text: The original text from which the chunk was extracted
            
        Returns:
            Metadata dictionary for the chunk
        """
        pass