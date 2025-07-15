from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.domain.models.tool_call_request import ToolCallRequest
from app.domain.models.tool_call_response import ToolCallResponse


class MCPToolHandler(ABC):
    """
    Abstract base class for MCP tool handlers.
    
    This interface defines the contract for classes that handle
    MCP tool calls and responses.
    """
    
    @abstractmethod
    async def handle_tool_call(self, request: ToolCallRequest) -> ToolCallResponse:
        """
        Handle an MCP tool call request.
        
        Args:
            request: The tool call request
            
        Returns:
            Tool call response
        """
        pass
    
    @abstractmethod
    def get_supported_tools(self) -> List[str]:
        """
        Get a list of supported tool names.
        
        Returns:
            List of supported tool names
        """
        pass
    
    @abstractmethod
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the JSON schema for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool schema dictionary or None if tool not found
        """
        pass