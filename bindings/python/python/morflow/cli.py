#!/usr/bin/env python3
"""
Morflow Python CLI Entrypoint.
Delegates command execution directly to the native Morflow Rust CLI implementation.
"""

import sys
from ._morflow import run_cli


def main():
    exit_code = run_cli(sys.argv[1:])
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
