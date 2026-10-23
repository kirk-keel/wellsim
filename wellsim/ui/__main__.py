"""Command-line entry point for launching the wellsim UI server."""

import sys
from wellsim.ui.server import run_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
