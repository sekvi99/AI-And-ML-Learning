from abc import ABC, abstractmethod
from pathlib import Path

from app.domain.models.document_info import DocumentInfo

class DocumentProcessor(ABC):
    """
    Abstract base class for document processing implementations.
    
    This interface defines the contract for classes that can extract text
    from various document formats (PDF, Word, etc.).
    """
    
    @abstractmethod
    async def extract_text(self, file_path: Path) -> str:
        """
        Extract text content from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentProcessingError: If extraction fails
        """
        pass
    
    @abstractmethod
    async def get_document_info(self, file_path: Path) -> DocumentInfo:
        """
        Get metadata information about a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document information and metadata
        """
        pass
    
    @abstractmethod
    def supports_format(self, file_path: Path) -> bool:
        """
        Check if this processor supports the given file format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if the format is supported, False otherwise
        """
        pass