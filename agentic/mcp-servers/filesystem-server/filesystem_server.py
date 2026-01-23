#!/usr/bin/env python3
"""
Filesystem MCP Server using the official MCP Python SDK (FastMCP)
Implements all tools from @modelcontextprotocol/server-filesystem
"""

import os
import re
import base64
import mimetypes
from pathlib import Path
from typing import List, Optional
import json

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Configuration
ALLOWED_DIRECTORIES = os.getenv('ALLOWED_DIRECTORIES', os.getcwd()).split(',')
ALLOWED_DIRECTORIES = [Path(d).resolve() for d in ALLOWED_DIRECTORIES]

# Create FastMCP server
# stateless_http=True makes it scalable for production (no session state)
mcp = FastMCP(
    name="filesystem-server",
    stateless_http=True,  # Recommended for production/Kubernetes
    json_response=True,  # Recommended for production/Kubernetes
    transport_security=TransportSecuritySettings(
        allowed_hosts=os.getenv('ALLOWED_HOSTS', '*').split(',')
    )
)


def validate_path(requested_path: str) -> Path:
    """Validate that path is within allowed directories"""
    absolute = Path(requested_path).resolve()
    
    is_allowed = any(
        str(absolute).startswith(str(allowed_dir))
        for allowed_dir in ALLOWED_DIRECTORIES
    )
    
    if not is_allowed:
        raise ValueError(f"Access denied: {requested_path} is outside allowed directories")
    
    return absolute


def get_mime_type(file_path: Path) -> str:
    """Get MIME type for a file"""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or 'application/octet-stream'


# ==================== TOOLS ====================

@mcp.tool(description="Read complete contents of a file as text. Supports head/tail options.")
def read_text_file(path: str, head: Optional[int] = None, tail: Optional[int] = None) -> str:
    """Read a text file with optional head/tail"""
    validated_path = validate_path(path)
    
    if not validated_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    
    content = validated_path.read_text(encoding='utf-8')
    
    if head and tail:
        raise ValueError("Cannot specify both head and tail")
    
    if head:
        lines = content.split('\n')
        content = '\n'.join(lines[:head])
    elif tail:
        lines = content.split('\n')
        content = '\n'.join(lines[-tail:])
    
    return content


@mcp.tool(description="Read an image or audio file and return as base64")
def read_media_file(path: str) -> str:
    """Read media file (image/audio) as base64"""
    validated_path = validate_path(path)
    
    if not validated_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    
    mime_type = get_mime_type(validated_path)
    
    if not (mime_type.startswith('image/') or mime_type.startswith('audio/')):
        raise ValueError(f"File is not an image or audio file: {mime_type}")
    
    with open(validated_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    
    return json.dumps({
        'mimeType': mime_type,
        'data': data,
        'path': str(validated_path)
    })


@mcp.tool(description="Read multiple files simultaneously")
def read_multiple_files(paths: List[str]) -> str:
    """Read multiple files at once"""
    results = []
    
    for file_path in paths:
        try:
            validated_path = validate_path(file_path)
            if validated_path.is_file():
                content = validated_path.read_text(encoding='utf-8')
                results.append({
                    'path': file_path,
                    'content': content,
                    'success': True
                })
            else:
                results.append({
                    'path': file_path,
                    'error': 'Not a file',
                    'success': False
                })
        except Exception as e:
            results.append({
                'path': file_path,
                'error': str(e),
                'success': False
            })
    
    return json.dumps(results, indent=2)


@mcp.tool(description="Create or overwrite a file with content")
def write_file(path: str, content: str) -> str:
    """Write content to a file"""
    validated_path = validate_path(path)
    
    # Create parent directories if needed
    validated_path.parent.mkdir(parents=True, exist_ok=True)
    
    validated_path.write_text(content, encoding='utf-8')
    
    return f'Successfully wrote to {path}'


@mcp.tool(description="Make line-based edits to a text file")
def edit_file(path: str, edits: List[dict], dry_run: bool = False) -> str:
    """
    Edit a file by replacing text.
    edits: List of dicts with 'oldText' and 'newText' keys
    """
    validated_path = validate_path(path)
    
    if not validated_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    
    content = validated_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    modified_lines = lines.copy()
    changes = []
    
    for edit in edits:
        old_text = edit['oldText']
        new_text = edit['newText']
        
        for i, line in enumerate(modified_lines):
            if old_text in line:
                modified_lines[i] = line.replace(old_text, new_text)
                changes.append(f"Line {i+1}: {line} -> {modified_lines[i]}")
    
    if not dry_run:
        validated_path.write_text('\n'.join(modified_lines), encoding='utf-8')
    
    prefix = '[DRY RUN] ' if dry_run else ''
    return f"{prefix}Changes:\n" + '\n'.join(changes)


@mcp.tool(description="Create a new directory or directories")
def create_directory(path: str) -> str:
    """Create a directory"""
    validated_path = validate_path(path)
    validated_path.mkdir(parents=True, exist_ok=True)
    return f'Successfully created directory {path}'


@mcp.tool(description="List contents of a directory")
def list_directory(path: str) -> str:
    """List directory contents"""
    validated_path = validate_path(path)
    
    if not validated_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    
    entries = []
    for item in sorted(validated_path.iterdir()):
        entries.append({
            'name': item.name,
            'type': 'directory' if item.is_dir() else 'file'
        })
    
    return json.dumps(entries, indent=2)


@mcp.tool(description="List directory contents with file sizes")
def list_directory_with_sizes(path: str) -> str:
    """List directory with sizes"""
    validated_path = validate_path(path)
    
    if not validated_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    
    entries = []
    for item in sorted(validated_path.iterdir()):
        entry = {
            'name': item.name,
            'type': 'directory' if item.is_dir() else 'file'
        }
        
        if item.is_file():
            entry['size'] = item.stat().st_size
        
        entries.append(entry)
    
    return json.dumps(entries, indent=2)


@mcp.tool(description="Get a recursive tree view of files and directories")
def directory_tree(
    path: str,
    exclude_patterns: Optional[List[str]] = None,
    max_depth: int = 2,   # 👈 sane default
) -> str:
    exclude_patterns = exclude_patterns or []
    root_path = validate_path(path)

    def build_tree(path: Path, depth: int) -> List[dict]:
        if depth > max_depth:
            return []

        result = []

        try:
            with os.scandir(path) as it:
                for entry in it:
                    if any(pat in entry.name for pat in exclude_patterns):
                        continue

                    node = {
                        "name": entry.name,
                        "type": "directory" if entry.is_dir(follow_symlinks=False) else "file",
                    }

                    if entry.is_dir(follow_symlinks=False):
                        node["children"] = build_tree(Path(entry.path), depth + 1)

                    result.append(node)
        except (PermissionError, FileNotFoundError):
            pass

        return result

    return json.dumps(build_tree(root_path, 0), indent=2)


@mcp.tool(description="Move or rename a file or directory")
def move_file(source: str, destination: str) -> str:
    """Move/rename file or directory"""
    source_path = validate_path(source)
    dest_path = validate_path(destination)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.rename(dest_path)
    
    return f'Successfully moved {source} to {destination}'


@mcp.tool(description="Search for files and directories by name pattern")
def search_files(path: str, pattern: str, exclude_patterns: Optional[List[str]] = None) -> str:
    """Search for files matching pattern"""
    exclude_patterns = exclude_patterns or []
    root_path = validate_path(path)

    exclude_re = re.compile("|".join(exclude_patterns)) if exclude_patterns else None
    matches = []

    try:
        for item in root_path.rglob(pattern):
            if exclude_re and exclude_re.search(item.name):
                continue
            matches.append(
                {
                    "path": str(item.relative_to(root_path)),
                    "type": "directory" if item.is_dir() else "file",
                }
            )
    except PermissionError:
        pass

    return json.dumps(matches, indent=2)


@mcp.tool(description="Get detailed information about a file or directory")
def get_file_info(path: str) -> str:
    """Get file/directory metadata"""
    validated_path = validate_path(path)
    
    if not validated_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    stat = validated_path.stat()
    
    info = {
        'path': str(validated_path),
        'name': validated_path.name,
        'type': 'directory' if validated_path.is_dir() else 'file',
        'size': stat.st_size if validated_path.is_file() else None,
        'created': stat.st_ctime,
        'modified': stat.st_mtime,
        'accessed': stat.st_atime,
        'permissions': oct(stat.st_mode)[-3:]
    }
    
    return json.dumps(info, indent=2)


@mcp.tool(description="List all directories the server is allowed to access")
def list_allowed_directories() -> str:
    """List allowed directories"""
    directories = [str(d) for d in ALLOWED_DIRECTORIES]
    return json.dumps(directories, indent=2)


# ==================== MAIN ====================

# Add health check endpoint using custom_route decorator
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint for Kubernetes probes"""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "healthy",
        "allowed_directories": [str(d) for d in ALLOWED_DIRECTORIES]
    })


if __name__ == "__main__":
    import sys
    import uvicorn
    
    print(f"\nFilesystem MCP Server (FastMCP + Streamable HTTP)", file=sys.stderr)
    print(f"Allowed directories: {', '.join(str(d) for d in ALLOWED_DIRECTORIES)}\n", file=sys.stderr)
    
    # Get configuration from environment
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"Starting server on {host}:{port}", file=sys.stderr)
    
    # Get the ASGI app from FastMCP and run it with uvicorn
    app = mcp.streamable_http_app()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )