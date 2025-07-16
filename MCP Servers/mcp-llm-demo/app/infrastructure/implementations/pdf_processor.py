import logging
from pathlib import Path
from typing import Dict, Any
import asyncio

import PyPDF2
import fitz  # PyMuPDF

from app.domain.interfaces import DocumentProcessor, DocumentProcessingError
from app.domain.models import DocumentInfo, ProcessingStatus

logger = logging.getLogger(__name__)


class PDFProcessor(DocumentProcessor):
    """
    Concrete implementation of DocumentProcessor for PDF files.
    
    This class handles PDF text extraction using PyMuPDF as the primary method
    with PyPDF2 as a fallback. It provides robust PDF processing with error
    handling and metadata extraction.
    """
    
    def __init__(self) -> None:
        """Initialize the PDF processor."""
        self.supported_extensions = {'.pdf'}
        logger.info("PDFProcessor initialized")
    
    async def extract_text(self, file_path: Path) -> str:
        """
        Extract text content from a PDF file.
        
        This method uses PyMuPDF (fitz) as the primary extraction method
        because it typically provides better text extraction quality than
        PyPDF2. If PyMuPDF fails, it falls back to PyPDF2.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content as a string
            
        Raises:
            DocumentProcessingError: If text extraction fails with both methods
        """
        if not file_path.exists():
            raise DocumentProcessingError(f"PDF file not found: {file_path}")
        
        if not self.supports_format(file_path):
            raise DocumentProcessingError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Starting text extraction from {file_path}")
        
        # Try PyMuPDF first (usually better quality)
        try:
            text = await self._extract_with_pymupdf(file_path)
            logger.info(f"Successfully extracted text using PyMuPDF from {file_path}")
            return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}, trying PyPDF2 fallback")
            
        # Fallback to PyPDF2
        try:
            text = await self._extract_with_pypdf2(file_path)
            logger.info(f"Successfully extracted text using PyPDF2 from {file_path}")
            return text
        except Exception as e:
            logger.error(f"Both extraction methods failed for {file_path}: {e}")
            raise DocumentProcessingError(f"Failed to extract text from PDF: {e}")
    
    async def _extract_with_pymupdf(self, file_path: Path) -> str:
        """
        Extract text using PyMuPDF (fitz).
        
        PyMuPDF generally provides better text extraction quality,
        especially for PDFs with complex layouts or embedded fonts.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            Exception: If PyMuPDF extraction fails
        """
        def _extract_sync() -> str:
            """Synchronous extraction to run in thread pool."""
            doc = fitz.open(str(file_path))
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Add page break marker for better chunking
                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}\n")
            
            doc.close()
            return "\n".join(text_parts)
        
        # Run synchronous extraction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _extract_sync)
    
    async def _extract_with_pypdf2(self, file_path: Path) -> str:
        """
        Extract text using PyPDF2 as fallback method.
        
        PyPDF2 is used as a fallback when PyMuPDF fails. It's generally
        less reliable but can work with some PDFs that PyMuPDF cannot handle.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            Exception: If PyPDF2 extraction fails
        """
        def _extract_sync() -> str:
            """Synchronous extraction to run in thread pool."""
            text_parts = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    
                    # Add page break marker for better chunking
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}\n")
            
            return "\n".join(text_parts)
        
        # Run synchronous extraction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _extract_sync)
    
    async def get_document_info(self, file_path: Path) -> DocumentInfo:
        """
        Get metadata information about a PDF document.
        
        This method extracts various metadata from the PDF including
        page count, title, author, and other document properties.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            DocumentInfo object with metadata
        """
        doc_info = DocumentInfo(
            path=file_path,
            exists=file_path.exists(),
            status=ProcessingStatus.PENDING
        )
        
        if not file_path.exists():
            doc_info.status = ProcessingStatus.FAILED
            doc_info.error_message = "File does not exist"
            return doc_info
        
        try:
            # Extract metadata using PyMuPDF
            metadata = await self._extract_pdf_metadata(file_path)
            doc_info.metadata = metadata
            doc_info.status = ProcessingStatus.COMPLETED
            
            logger.info(f"Successfully extracted metadata from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            doc_info.status = ProcessingStatus.FAILED
            doc_info.error_message = str(e)
        
        return doc_info
    
    async def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from PDF using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        def _extract_sync() -> Dict[str, Any]:
            """Synchronous metadata extraction."""
            doc = fitz.open(str(file_path))
            
            metadata = {
                'page_count': len(doc),
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'file_size': file_path.stat().st_size,
                'encrypted': doc.is_encrypted,
                'pdf_version': doc.pdf_version(),
            }
            
            doc.close()
            return metadata
        
        # Run synchronous metadata extraction in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _extract_sync)
    
    def supports_format(self, file_path: Path) -> bool:
        """
        Check if this processor supports the given file format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if the file is a PDF, False otherwise
        """
        return file_path.suffix.lower() in self.supported_extensions


class MultiFormatDocumentProcessor(DocumentProcessor):
    """
    Document processor that can handle multiple file formats.
    
    This class acts as a dispatcher that routes different file formats
    to their appropriate processors. It can be easily extended to support
    additional document formats.
    """
    
    def __init__(self) -> None:
        """Initialize the multi-format processor with available processors."""
        self.processors = {
            '.pdf': PDFProcessor(),
        }
        logger.info("MultiFormatDocumentProcessor initialized")
    
    async def extract_text(self, file_path: Path) -> str:
        """
        Extract text using the appropriate processor for the file format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentProcessingError: If no processor supports the format
        """
        processor = self._get_processor_for_file(file_path)
        return await processor.extract_text(file_path)
    
    async def get_document_info(self, file_path: Path) -> DocumentInfo:
        """
        Get document info using the appropriate processor for the file format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document information
        """
        processor = self._get_processor_for_file(file_path)
        return await processor.get_document_info(file_path)
    
    def supports_format(self, file_path: Path) -> bool:
        """
        Check if any processor supports the given file format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if the format is supported, False otherwise
        """
        return file_path.suffix.lower() in self.processors
    
    def _get_processor_for_file(self, file_path: Path) -> DocumentProcessor:
        """
        Get the appropriate processor for a file format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document processor for the file format
            
        Raises:
            DocumentProcessingError: If no processor supports the format
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.processors:
            supported_formats = list(self.processors.keys())
            raise DocumentProcessingError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {supported_formats}"
            )
        
        return self.processors[file_extension]
    
    def add_processor(self, extension: str, processor: DocumentProcessor) -> None:
        """
        Add a new processor for a specific file extension.
        
        This method allows extending the multi-format processor with
        additional document format support at runtime.
        
        Args:
            extension: File extension (e.g., '.docx', '.txt')
            processor: Document processor instance for the format
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        self.processors[extension.lower()] = processor
        logger.info(f"Added processor for {extension} format")
    
    def get_supported_formats(self) -> list[str]:
        """
        Get a list of all supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return list(self.processors.keys())
