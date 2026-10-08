"""Convenience wrapper for the installed bangla-absa command."""

import sys

from bangla_absa.cli import main

if __name__ == "__main__":
    main(["repeat", *sys.argv[1:]])
