"""
mcp_server.py — minimal FastMCP test server.

Exposes a single tool: read_file(filename)
which reads from the ./files/ directory and returns the contents.

Run with:
    pip install fastmcp
    python mcp_server.py
"""

import os
from pathlib import Path

from fastmcp import FastMCP

FILES_DIR = Path(__file__).parent / "files"

mcp = FastMCP("file-reader")


@mcp.tool()
def read_file(filename: str) -> str:
    """Read and return the contents of a .txt file from the files/ directory.

    Args:
        filename: Name of the file to read, e.g. "tariffs.txt".
                  Do not include a path — only the filename.

    Returns the file contents as a string.
    """
    safe_name = Path(filename).name
    file_path = FILES_DIR / safe_name

    if not file_path.exists():
        available = [f.name for f in FILES_DIR.glob("*.txt")]
        return f"File '{safe_name}' not found. Available files: {available}"

    return file_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8001)