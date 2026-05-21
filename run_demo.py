#!/usr/bin/env python3
"""BreathCoach demo launcher — start the inference + web server.

This script makes the project runnable straight from a clone with no install
step: it puts ``src/`` on the import path and hands off to the server's main().

    python run_demo.py                 # auto-discovers the bundled backbone
    python run_demo.py --port 9000     # any serve.py flag passes through

It exists so you don't have to `pip install -e .` (editable installs can be
fragile when the project lives on a path with spaces or under a synced folder).
If you HAVE installed the package, `python -m nanobreath.deployment.serve` works
identically.
"""
import os
import sys

# Put the package's src/ directory on sys.path so `import nanobreath` resolves
# regardless of whether the package is pip-installed.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from nanobreath.deployment.serve import main  # noqa: E402

if __name__ == "__main__":
    main()
