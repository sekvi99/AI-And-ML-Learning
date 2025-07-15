from enum import Enum


class ProcessingStatus(str, Enum):
    """Enumeration of possible processing states for documents."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"