class DocumentProcessingError(Exception):
    """Exception raised when document processing fails."""
    pass


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""
    pass


class SearchError(Exception):
    """Exception raised when search operations fail."""
    pass


class KnowledgeBaseError(Exception):
    """Exception raised when knowledge base operations fail."""
    pass


class MCPToolError(Exception):
    """Exception raised when MCP tool operations fail."""
    pass