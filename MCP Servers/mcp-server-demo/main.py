# server.py
from mcp.server.fastmcp import FastMCP
import os

NOTES_FILE = os.path.join(os.path.dirname(__file__), "notes.txt")


def ensure_file():
    """Ensure the notes file exists."""
    if not os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "w") as f:
            f.write("")

# Create an MCP server
mcp = FastMCP("AI Sticky Notes")

@mcp.tool()
def add_note(message: str) -> str:
    """
    Append a note to the notes file.

    Args:
        message (str): The note message to append.

    Returns:
        str: Confirmation message indicating the note was added.
    """
    ensure_file()
    with open(NOTES_FILE, "a") as f:
        f.write(f"{message}\n")
    return f"Note added: {message}"

@mcp.tool()
def read_notes() -> str:
    """
    Read all notes from the notes file.

    Returns:
        str: All notes as a single string.
    """
    ensure_file()
    with open(NOTES_FILE, "r") as f:
        notes = f.read().strip()
    return notes if notes else "No notes found."

@mcp.resource("notes://latest")
def get_latest_note() -> str:
    """
    Get the latest note from the notes file.

    Returns:
        str: The latest note or a message if no notes exist.
    """
    ensure_file()
    with open(NOTES_FILE, "r") as f:
        notes = f.readlines()
    return notes[-1].strip() if notes else "No notes found."

@mcp.prompt()
def note_summary_prompt(notes: str) -> str:
    """
    Generate a prompt to summarize the current notes.

    Args:
        notes (str): The current notes to summarize.

    Returns:
        str: A prompt for summarizing the notes.
    """
    ensure_file()
    with open(NOTES_FILE, "r") as f:
        notes = f.read().strip()
    if not notes:
        return "No notes available to summarize."
    
    return f"Summarize the current notes:\n{notes}"
