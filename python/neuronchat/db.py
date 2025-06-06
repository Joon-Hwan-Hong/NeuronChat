"""Interaction database utilities."""

from __future__ import annotations

from typing import Dict

import json
import os


def update_interactionDB(path: str) -> Dict[str, dict]:
    """Load interaction database from ``path``.

    The R package ships with serialized data tables. Here we expect ``path``
    to be a JSON file for simplicity.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as fh:
        db = json.load(fh)
    return db
