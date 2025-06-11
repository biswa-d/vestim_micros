#!/usr/bin/env python
"""
Simple server script for VEstim.
Runs a FastAPI server that manages machine learning model training jobs.
"""
import argparse
import os
import sys
import uvicorn

def main():
    """
    Main entry point for running the VEstim backend server.
    """
    parser = argparse.ArgumentParser(description="VEstim Backend Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()

    print(f"Starting VEstim server on {args.host}:{args.port}...")
    
    try:
        # Run the server
        uvicorn.run(
            "vestim.backend.src.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Error: Port {args.port} is already in use.")
            print(f"This likely means the server is already running at {args.host}:{args.port}")
            print("If you need to restart the server, use 'vestim stop' first, then start it again.")
        else:
            print(f"Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
