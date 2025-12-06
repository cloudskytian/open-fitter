import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json


def load_avatar_data(filepath: str) -> dict:
    """Load and parse avatar data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load avatar data: {str(e)}")
