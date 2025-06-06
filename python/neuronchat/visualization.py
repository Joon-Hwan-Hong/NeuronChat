"""Visualization utilities for NeuronChat."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def netVisual_circle_neuron(matrix: np.ndarray, labels: Iterable[str], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Draw network as a circle plot. Placeholder implementation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    coords = np.c_[np.cos(angles), np.sin(angles)]
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, labels[i], ha="center", va="center")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] > 0:
                ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], "k-", alpha=0.3)
    ax.set_axis_off()
    return ax


def heatmap_single(matrix: np.ndarray, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """Simple heatmap wrapper using seaborn."""
    if ax is None:
        fig, ax = plt.subplots()
    sns.heatmap(matrix, ax=ax, **kwargs)
    return ax
