"""
agent_tools.py

Contains tool functions for document update and save operations.
"""

from langchain_core.tools import tool

# Global variable to hold the document content
DOCUMENT_CONTENT: str = ""

@tool
def update(content: str) -> str:
    """
    Updates the document with provided content.

    Args:
        content (str): Content that updates the document

    Returns:
        str: Information that content has been updated with new document content
    """
    global DOCUMENT_CONTENT
    DOCUMENT_CONTENT = content
    return f"Document has been updated successfully! The current content is:\n{DOCUMENT_CONTENT}"

@tool
def save(filename: str) -> str:
    """
    Saves the current document to a text file and finish the process.

    Args:
        filename (str): Name for the text file

    Returns:
        str: Status message about saving the document
    """
    global DOCUMENT_CONTENT
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    try:
        with open(filename, 'w') as file:
            file.write(DOCUMENT_CONTENT)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    except Exception as e:
        return f"Error saving document: {str(e)}"