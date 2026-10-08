"""Convenience wrapper for the installed bangla-absa command."""

import sys

from bangla_absa.cli import main

if __name__ == "__main__":
    main(["export-tables", *sys.argv[1:]])
