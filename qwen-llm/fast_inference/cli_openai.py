#!/usr/bin/env python3
"""
OpenAI-Compatible API Server CLI

Command-line interface for running the OpenAI-compatible API server.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path to import fast_inference
sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_inference.core.engine.openai_api_server import main

if __name__ == "__main__":
    main()
