"""Interaction database utilities.

The Python API expects interaction databases to be provided as JSON files.  A
database is a mapping from the interaction name to an object with the
following keys::

    {
        "interaction_name": str,
        "lig_contributor": [str, ...],
        "receptor_subunit": [str, ...],
        "lig_contributor_group": [int, ...],
        "lig_contributor_coeff": [float, ...],
        "receptor_subunit_group": [int, ...],
        "receptor_subunit_coeff": [float, ...],
        "targets_up": [str, ...],
        "targets_down": [str, ...],
        "activator": [str, ...],
        "inhibitor": [str, ...],
        "interactors": [str, ...],
        "interaction_type": str,
        "ligand_type": str,
    }

The ``scripts/convert_rda.py`` helper converts the ``.rda`` databases shipped
with the R package into this JSON structure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import os


def load_interactionDB(path: str) -> Dict[str, dict]:
    """Load interaction database from ``path``.

    The R package ships with serialized data tables. Here we expect ``path``
    to be a JSON file for simplicity.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path) as fh:
        db = json.load(fh)

    return db


def update_interactionDB(
    DB: Dict[str, Any],
    interaction_name: str,
    lig_contributor: List[str],
    receptor_subunit: List[str],
    interaction_type: str = "user_defined",
    ligand_type: str = "user_defined",
    lig_contributor_group: Optional[List[int]] = None,
    lig_contributor_coeff: Optional[List[float]] = None,
    receptor_subunit_group: Optional[List[int]] = None,
    receptor_subunit_coeff: Optional[List[float]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Add or modify an interaction entry in ``DB``.

    Parameters mirror :func:`update_interactionDB` in the R package.  The
    function validates the contributor groups and coefficient vectors before
    inserting ``interaction_name`` into ``DB``.  If ``interaction_name``
    already exists, ``overwrite`` must be ``True`` to replace it.
    """

    if interaction_name in DB and not overwrite:
        raise ValueError(f"{interaction_name!r} already exists in DB")

    if lig_contributor_group is None:
        lig_contributor_group = [1] * len(lig_contributor)
        lig_contributor_coeff = [1]

    if receptor_subunit_group is None:
        receptor_subunit_group = [1] * len(receptor_subunit)
        receptor_subunit_coeff = [1]

    if len(lig_contributor_group) != len(lig_contributor) or len(receptor_subunit_group) != len(receptor_subunit):
        raise ValueError(
            "`lig_contributor_group` or `receptor_subunit_group` length mismatch"
        )

    if lig_contributor_coeff is None:
        lig_contributor_coeff = [1] * len(set(lig_contributor_group))
    if receptor_subunit_coeff is None:
        receptor_subunit_coeff = [1] * len(set(receptor_subunit_group))

    if len(lig_contributor_coeff) != len(set(lig_contributor_group)) or len(receptor_subunit_coeff) != len(set(receptor_subunit_group)):
        raise ValueError(
            "`lig_contributor_coeff` or `receptor_subunit_coeff` length mismatch"
        )

    entry = {
        "interaction_name": interaction_name,
        "lig_contributor": lig_contributor,
        "receptor_subunit": receptor_subunit,
        "lig_contributor_group": lig_contributor_group,
        "lig_contributor_coeff": lig_contributor_coeff,
        "receptor_subunit_group": receptor_subunit_group,
        "receptor_subunit_coeff": receptor_subunit_coeff,
        "targets_up": [],
        "targets_down": [],
        "activator": [],
        "inhibitor": [],
        "interactors": [],
        "interaction_type": interaction_type,
        "ligand_type": ligand_type,
    }

    DB[interaction_name] = entry
    return DB
