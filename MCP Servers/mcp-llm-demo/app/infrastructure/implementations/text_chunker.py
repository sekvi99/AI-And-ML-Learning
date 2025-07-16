import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.domain.interfaces import TextChunker
from app.domain.models import TextChunk, ChunkingConfig

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Helper class for text splitting operations."""
    text: str
    start_pos: int
    end_pos: int
    page_number: Optional[int] = None


class SentenceBasedChunker(TextChunker):
    """
    Text chunker that splits text based on sentence boundaries.
    
    This implementation attempts to create chunks that respect sentence
    boundaries while staying within the target chunk size. It provides
    better semantic coherence compared to fixed-size chunking.
    """
    
    def __init__(self) -> None:
        """Initialize the sentence-based chunker."""
        # Regular expression for sentence splitting
        # This pattern looks for sentence-ending punctuation followed by whitespace
        # and a capital letter or end of string
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$',
            re.MULTILINE
        )
        
        # Pattern to identify page breaks in text
        self.page_break_pattern = re.compile(
            r'--- Page (\d+) ---',
            re.MULTILINE
        )
        
        logger.info("SentenceBasedChunker initialized")
    
    def chunk_text(self, text: str, config: ChunkingConfig) -> List[TextChunk]:
        """
        Split text into chunks based on sentence boundaries.
        
        This method creates chunks that respect sentence boundaries while
        attempting to stay within the target chunk size. It also handles
        overlap between chunks and maintains page number information.
        
        Args:
            text: The text to be chunked
            config: Configuration for chunking strategy
            
        Returns:
            List of TextChunk objects
        """
        logger.info(f"Starting text chunking with config: {config}")
        
        # Clean and normalize the text
        cleaned_text = self._clean_text(text)
        
        # Extract page information
        page_info = self._extract_page_info(cleaned_text)
        
        # Remove page markers from text for chunking
        clean_text = self.page_break_pattern.sub('', cleaned_text)
        
        # Split into sentences
        sentences = self._split_into_sentences(clean_text)
        
        # Create chunks from sentences
        chunks = self._create_chunks_from_sentences(
            sentences, config, page_info
        )
        
        # Filter out chunks that are too small
        filtered_chunks = [
            chunk for chunk in chunks 
            if len(chunk.text) >= config.min_chunk_size
        ]
        
        logger.info(f"Created {len(filtered_chunks)} chunks from {len(sentences)} sentences")
        
        return filtered_chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for better chunking.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_page_info(self, text: str) -> Dict[int, int]:
        """
        Extract page number information from text.
        
        Args:
            text: Text containing page markers
            
        Returns:
            Dictionary mapping character positions to page numbers
        """
        page_info = {}
        current_page = 1
        
        for match in self.page_break_pattern.finditer(text):
            page_num = int(match.group(1))
            page_info[match.start()] = page_num
            current_page = page_num
        
        return page_info
    
    def _split_into_sentences(self, text: str) -> List[SplitResult]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of SplitResult objects representing sentences
        """
        sentences = []
        last_end = 0
        
        for match in self.sentence_pattern.finditer(text):
            sentence_text = text[last_end:match.end()].strip()
            
            if sentence_text:
                sentences.append(SplitResult(
                    text=sentence_text,
                    start_pos=last_end,
                    end_pos=match.end()
                ))
            
            last_end = match.end()
        
        # Add the last sentence if it doesn't end with punctuation
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                sentences.append(SplitResult(
                    text=remaining_text,
                    start_pos=last_end,
                    end_pos=len(text)
                ))
        
        return sentences
    
    def _create_chunks_from_sentences(
        self, 
        sentences: List[SplitResult], 
        config: ChunkingConfig,
        page_info: Dict[int, int]
    ) -> List[TextChunk]:
        """
        Create chunks from sentences while respecting size limits.
        
        Args:
            sentences: List of sentence split results
            config: Chunking configuration
            page_info: Page number information
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        chunk_id = 0
        
        i = 0
        while i < len(sentences):
            chunk_text = ""
            chunk_start = sentences[i].start_pos
            chunk_end = sentences[i].end_pos
            chunk_sentences = []
            
            # Build chunk by adding sentences until size limit is reached
            while i < len(sentences):
                sentence = sentences[i]
                potential_text = chunk_text + " " + sentence.text if chunk_text else sentence.text
                
                # Check if adding this sentence would exceed the limit
                if len(potential_text) > config.chunk_size and chunk_text:
                    break
                
                chunk_text = potential_text
                chunk_end = sentence.end_pos
                chunk_sentences.append(sentence)
                i += 1
            
            # Determine page number for this chunk
            page_number = self._get_page_number_for_position(chunk_start, page_info)
            
            # Create the chunk
            chunk = TextChunk(
                id=chunk_id,
                text=chunk_text,
                page_number=page_number,
                start_position=chunk_start,
                end_position=chunk_end,
                metadata=self._create_chunk_metadata(chunk_sentences, config)
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Handle overlap by backing up some sentences
            if config.overlap > 0 and i < len(sentences):
                overlap_chars = 0
                overlap_sentences = 0
                
                # Find how many sentences to include in overlap
                for j in range(len(chunk_sentences) - 1, -1, -1):
                    overlap_chars += len(chunk_sentences[j].text)
                    overlap_sentences += 1
                    
                    if overlap_chars >= config.overlap:
                        break
                
                # Back up for overlap
                i -= overlap_sentences
                i = max(0, i)
        
        return chunks
    
    def _get_page_number_for_position(
        self, 
        position: int, 
        page_info: Dict[int, int]
    ) -> Optional[int]:
        """
        Get the page number for a given text position.
        
        Args:
            position: Character position in text
            page_info: Page number mapping
            
        Returns:
            Page number or None if not found
        """
        if not page_info:
            return None
        
        # Find the closest page marker before this position
        closest_page = None
        closest_pos = -1
        
        for pos, page_num in page_info.items():
            if pos <= position and pos > closest_pos:
                closest_pos = pos
                closest_page = page_num
        
        return closest_page
    
    def _create_chunk_metadata(
        self, 
        sentences: List[SplitResult], 
        config: ChunkingConfig
    ) -> Dict[str, Any]:
        """
        Create metadata for a chunk.
        
        Args:
            sentences: List of sentences in the chunk
            config: Chunking configuration
            
        Returns:
            Metadata dictionary
        """
        return {
            'sentence_count': len(sentences),
            'chunk_strategy': config.split_strategy,
            'target_size': config.chunk_size,
            'overlap': config.overlap,
            'actual_size': sum(len(s.text) for s in sentences)
        }
    
    def get_chunk_metadata(self, chunk: TextChunk, original_text: str) -> Dict[str, Any]:
        """
        Extract additional metadata for a text chunk.
        
        Args:
            chunk: The text chunk
            original_text: The original text from which the chunk was extracted
            
        Returns:
            Enhanced metadata dictionary
        """
        metadata = chunk.metadata.copy()
        
        # Add statistics about the chunk
        metadata.update({
            'word_count': len(chunk.text.split()),
            'character_count': len(chunk.text),
            'has_page_info': chunk.page_number is not None,
            'chunk_id': chunk.id,
        })
        
        # Add context information if available
        if chunk.start_position is not None and chunk.end_position is not None:
            # Calculate relative position in document
            total_length = len(original_text)
            relative_start = chunk.start_position / total_length
            relative_end = chunk.end_position / total_length
            
            metadata.update({
                'relative_position_start': relative_start,
                'relative_position_end': relative_end,
                'document_coverage': relative_end - relative_start
            })
        
        return metadata


class FixedSizeChunker(TextChunker):
    """
    Text chunker that splits text into fixed-size chunks.
    
    This is a simpler chunking strategy that splits text into chunks of
    a fixed character size. It's faster but may break sentences in the middle.
    """
    
    def __init__(self) -> None:
        """Initialize the fixed-size chunker."""
        logger.info("FixedSizeChunker initialized")
    
    def chunk_text(self, text: str, config: ChunkingConfig) -> List[TextChunk]:
        """
        Split text into fixed-size chunks.
        
        Args:
            text: The text to be chunked
            config: Configuration for chunking strategy
            
        Returns:
            List of TextChunk objects
        """
        logger.info(f"Starting fixed-size chunking with config: {config}")
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        chunks = []
        chunk_id = 0
        position = 0
        
        while position < len(cleaned_text):
            # Calculate chunk boundaries
            chunk_start = position
            chunk_end = min(position + config.chunk_size, len(cleaned_text))
            
            # Extract chunk text
            chunk_text = cleaned_text[chunk_start:chunk_end]
            
            # Skip chunks that are too small
            if len(chunk_text) < config.min_chunk_size:
                break
            
            # Create chunk
            chunk = TextChunk(
                id=chunk_id,
                text=chunk_text,
                start_position=chunk_start,
                end_position=chunk_end,
                metadata=self._create_chunk_metadata(chunk_text, config)
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move position forward, accounting for overlap
            position = chunk_end - config.overlap
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _create_chunk_metadata(self, text: str, config: ChunkingConfig) -> Dict[str, Any]:
        """Create metadata for a chunk."""
        return {
            'chunk_strategy': 'fixed_size',
            'target_size': config.chunk_size,
            'actual_size': len(text),
            'word_count': len(text.split()),
            'overlap': config.overlap
        }
    
    def get_chunk_metadata(self, chunk: TextChunk, original_text: str) -> Dict[str, Any]:
        """Extract additional metadata for a text chunk."""
        metadata = chunk.metadata.copy()
        
        # Add basic statistics
        metadata.update({
            'character_count': len(chunk.text),
            'chunk_id': chunk.id,
        })
        
        return metadata


class ChunkerFactory:
    """
    Factory class for creating text chunkers.
    
    This factory provides a centralized way to create different types
    of text chunkers based on configuration.
    """
    
    @staticmethod
    def create_chunker(strategy: str) -> TextChunker:
        """
        Create a text chunker based on the specified strategy.
        
        Args:
            strategy: Chunking strategy name
            
        Returns:
            TextChunker instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        chunkers = {
            'sentence': SentenceBasedChunker,
            'fixed_size': FixedSizeChunker,
            'paragraph': SentenceBasedChunker,  # Can be extended later
        }
        
        if strategy not in chunkers:
            raise ValueError(
                f"Unsupported chunking strategy: {strategy}. "
                f"Available strategies: {list(chunkers.keys())}"
            )
        
        return chunkers[strategy]()
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """
        Get a list of available chunking strategies.
        
        Returns:
            List of available strategy names
        """
        return ['sentence', 'fixed_size', 'paragraph']