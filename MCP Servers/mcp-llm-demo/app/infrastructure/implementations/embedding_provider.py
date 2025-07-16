import logging
import asyncio
from typing import List, Optional, Dict, Any
import numpy as np

from sentence_transformers import SentenceTransformer
import torch

from app.domain.interfaces import EmbeddingProvider, EmbeddingError
from app.domain.models import EmbeddingConfig

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Embedding provider using Sentence Transformers library.
    
    This implementation uses the sentence-transformers library to generate
    semantic embeddings for text. It supports various pre-trained models
    and provides efficient batch processing.
    """
    
    def __init__(self) -> None:
        """Initialize the sentence transformer provider."""
        self.model: Optional[SentenceTransformer] = None
        self.config: Optional[EmbeddingConfig] = None
        self.device: str = "cpu"
        self._embedding_dimension: Optional[int] = None
        
        logger.info("SentenceTransformerProvider initialized")
    
    async def initialize(self, config: EmbeddingConfig) -> None:
        """
        Initialize the embedding provider with the given configuration.
        
        This method loads the specified sentence transformer model and
        configures it according to the provided settings. The model loading
        is done asynchronously to avoid blocking the main thread.
        
        Args:
            config: Configuration for the embedding model
            
        Raises:
            EmbeddingError: If model initialization fails
        """
        logger.info(f"Initializing embedding provider with model: {config.model_name}")
        
        try:
            self.config = config
            
            # Determine device
            self.device = self._determine_device(config.device)
            logger.info(f"Using device: {self.device}")
            
            # Load model in thread pool to avoid blocking
            self.model = await self._load_model_async(config)
            
            # Configure model settings
            self._configure_model(config)
            
            # Get embedding dimension
            self._embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(
                f"Successfully initialized {config.model_name} "
                f"with {self._embedding_dimension}D embeddings"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider: {e}")
            raise EmbeddingError(f"Failed to initialize embedding model: {e}")
    
    def _determine_device(self, device_config: str) -> str:
        """
        Determine the appropriate device for the model.
        
        Args:
            device_config: Device configuration ('cpu', 'cuda', 'auto')
            
        Returns:
            Device string to use
        """
        if device_config == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device_config == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        else:
            return device_config
    
    async def _load_model_async(self, config: EmbeddingConfig) -> SentenceTransformer:
        """
        Load the sentence transformer model asynchronously.
        
        Args:
            config: Embedding configuration
            
        Returns:
            Loaded SentenceTransformer model
        """
        def _load_model() -> SentenceTransformer:
            """Synchronous model loading function."""
            return SentenceTransformer(
                config.model_name,
                device=self.device
            )
        
        # Run model loading in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _load_model)
    
    def _configure_model(self, config: EmbeddingConfig) -> None:
        """
        Configure model settings based on the configuration.
        
        Args:
            config: Embedding configuration
        """
        if self.model is None:
            raise EmbeddingError("Model not initialized")
        
        # Set maximum sequence length
        self.model.max_seq_length = config.max_length
        
        # Additional model configuration can be added here
        logger.info(f"Model configured with max_length={config.max_length}")
    
    async def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        This method processes texts in batches to efficiently generate
        embeddings. It handles large lists by processing them in chunks
        to avoid memory issues.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
            
        Raises:
            EmbeddingError: If encoding fails
        """
        if self.model is None or self.config is None:
            raise EmbeddingError("Embedding provider not initialized")
        
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts")
        
        try:
            # Process in batches to handle large inputs
            batch_size = self.config.batch_size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self._encode_batch(batch)
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batch results
            embeddings = np.vstack(all_embeddings)
            
            # Normalize embeddings if requested
            if self.config.normalize_embeddings:
                embeddings = self._normalize_embeddings(embeddings)
            
            logger.info(f"Successfully encoded {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise EmbeddingError(f"Failed to encode texts: {e}")
    
    async def _encode_batch(self, batch: List[str]) -> np.ndarray:
        """
        Encode a batch of texts.
        
        Args:
            batch: List of texts to encode
            
        Returns:
            Numpy array of embeddings for the batch
        """
        def _encode_sync() -> np.ndarray:
            """Synchronous encoding function."""
            return self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        
        # Run encoding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _encode_sync)
    
    async def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        This method is optimized for single query encoding and applies
        the same processing as batch encoding but for a single text.
        
        Args:
            query: Search query string
            
        Returns:
            Numpy array representing the query embedding
            
        Raises:
            EmbeddingError: If encoding fails
        """
        if self.model is None or self.config is None:
            raise EmbeddingError("Embedding provider not initialized")
        
        if not query.strip():
            raise EmbeddingError("Query cannot be empty")
        
        logger.debug(f"Encoding query: {query[:50]}...")
        
        try:
            # Encode single query
            embeddings = await self.encode_text([query])
            
            # Return the first (and only) embedding
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise EmbeddingError(f"Failed to encode query: {e}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit vectors.
        
        Args:
            embeddings: Raw embeddings array
            
        Returns:
            Normalized embeddings array
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this provider.
        
        Returns:
            Embedding dimension
            
        Raises:
            EmbeddingError: If provider not initialized
        """
        if self._embedding_dimension is None:
            raise EmbeddingError("Embedding provider not initialized")
        
        return self._embedding_dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None or self.config is None:
            return {"status": "not_initialized"}
        
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "embedding_dimension": self._embedding_dimension,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "normalize_embeddings": self.config.normalize_embeddings,
            "status": "initialized"
        }


class CachedEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider with caching capabilities.
    
    This wrapper adds caching functionality to any embedding provider
    to avoid recomputing embeddings for the same texts.
    """
    
    def __init__(self, base_provider: EmbeddingProvider, cache_size: int = 1000) -> None:
        """
        Initialize the cached embedding provider.
        
        Args:
            base_provider: The underlying embedding provider
            cache_size: Maximum number of embeddings to cache
        """
        self.base_provider = base_provider
        self.cache_size = cache_size
        self.text_cache: Dict[str, np.ndarray] = {}
        self.cache_order: List[str] = []
        
        logger.info(f"CachedEmbeddingProvider initialized with cache_size={cache_size}")
    
    async def initialize(self, config: EmbeddingConfig) -> None:
        """Initialize the base provider."""
        await self.base_provider.initialize(config)
    
    async def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings with caching support.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Separate cached and uncached texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.text_cache:
                cached_embeddings[i] = self.text_cache[text]
                # Update cache order
                self.cache_order.remove(text)
                self.cache_order.append(text)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Encode uncached texts
        if uncached_texts:
            new_embeddings = await self.base_provider.encode_text(uncached_texts)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                self._add_to_cache(text, embedding)
        
        # Reconstruct full embeddings array
        all_embeddings = []
        uncached_idx = 0
        
        for i in range(len(texts)):
            if i in cached_embeddings:
                all_embeddings.append(cached_embeddings[i])
            else:
                all_embeddings.append(new_embeddings[uncached_idx])
                uncached_idx += 1
        
        return np.array(all_embeddings)
    
    async def encode_query(self, query: str) -> np.ndarray:
        """Encode query with caching."""
        embeddings = await self.encode_text([query])
        return embeddings[0]
    
    def _add_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """
        Add an embedding to the cache.
        
        Args:
            text: Text string
            embedding: Corresponding embedding
        """
        # Remove oldest entry if cache is full
        if len(self.text_cache) >= self.cache_size:
            oldest_text = self.cache_order.pop(0)
            del self.text_cache[oldest_text]
        
        # Add new entry
        self.text_cache[text] = embedding
        self.cache_order.append(text)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension from base provider."""
        return self.base_provider.get_embedding_dimension()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self.text_cache),
            "max_cache_size": self.cache_size,
            "cache_hit_rate": self._calculate_hit_rate()
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)."""
        # This is a simplified calculation
        # In practice, you'd track hits and misses
        return 0.0
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.text_cache.clear()
        self.cache_order.clear()
        logger.info("Embedding cache cleared")


class EmbeddingProviderFactory:
    """
    Factory for creating embedding providers.
    
    This factory provides a centralized way to create different types
    of embedding providers based on configuration.
    """
    
    @staticmethod
    def create_provider(
        provider_type: str = "sentence_transformer",
        use_cache: bool = True,
        cache_size: int = 1000
    ) -> EmbeddingProvider:
        """
        Create an embedding provider.
        
        Args:
            provider_type: Type of provider to create
            use_cache: Whether to use caching
            cache_size: Size of the cache if caching is enabled
            
        Returns:
            EmbeddingProvider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type == "sentence_transformer":
            base_provider = SentenceTransformerProvider()
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        if use_cache:
            return CachedEmbeddingProvider(base_provider, cache_size)
        else:
            return base_provider
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """
        Get a list of available embedding provider types.
        
        Returns:
            List of provider type names
        """
        return ["sentence_transformer"]