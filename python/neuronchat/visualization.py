"""Visualization utilities for NeuronChat."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import gridspec
import numpy as np
import seaborn as sns

__all__ = [
    "sc_palette",
    "netVisual_circle_neuron",
    "heatmap_single",
]


def sc_palette(n: int) -> list[str]:
    """Color palette used by CellChat's ``scPalette`` function."""

    color_space = [
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#F29403",
        "#F781BF",
        "#BC9DCC",
        "#A65628",
        "#54B0E4",
        "#222F75",
        "#1B9E77",
        "#B2DF8A",
        "#E3BE00",
        "#FB9A99",
        "#E7298A",
        "#910241",
        "#00CDD1",
        "#A6CEE3",
        "#CE1261",
        "#5E4FA2",
        "#8CA77B",
        "#00441B",
        "#DEDC00",
        "#B3DE69",
        "#8DD3C7",
        "#999999",
    ]

    if n <= len(color_space):
        return color_space[:n]
    palette = sns.color_palette(color_space, n)
    return [sns.utils.rgb2hex(c) for c in palette]


def netVisual_circle_neuron(
    matrix: np.ndarray,
    labels: Iterable[str],
    *,
    group: Optional[Mapping[str, str]] = None,
    color_use: Optional[Iterable[str]] = None,
    vertex_weight: Optional[np.ndarray] = None,
    vertex_label_size: float = 10,
    edge_width_max: float = 5.0,
    edge_alpha: float = 0.6,
    show_edge_labels: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw a circle plot for a weighted network."""

    labels = list(labels)
    n = len(labels)
    matrix = np.asarray(matrix)
    if matrix.shape != (n, n):
        raise ValueError("matrix shape and labels length mismatch")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = np.c_[np.cos(angles), np.sin(angles)]

    if vertex_weight is None:
        vertex_weight = np.ones(n)
    vertex_weight = np.asarray(vertex_weight)

    if group is None:
        group = {lab: lab for lab in labels}

    groups = [group.get(lab, lab) for lab in labels]
    unique_groups = list(dict.fromkeys(groups))

    if color_use is None:
        color_use = sc_palette(len(unique_groups))
    color_map = {g: c for g, c in zip(unique_groups, color_use)}

    max_edge = matrix.max() if matrix.size > 0 else 1.0

    for i, (x, y) in enumerate(coords):
        ax.scatter(
            x,
            y,
            s=100 * vertex_weight[i],
            color=color_map[groups[i]],
            edgecolor="black",
            zorder=3,
        )
        ax.text(x, y, labels[i], ha="center", va="center", fontsize=vertex_label_size)

    for i in range(n):
        for j in range(n):
            w = matrix[i, j]
            if w <= 0:
                continue
            width = (w / max_edge) * edge_width_max
            arrow = FancyArrowPatch(
                coords[i],
                coords[j],
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=width,
                color=color_map[groups[i]],
                alpha=edge_alpha,
                zorder=1,
            )
            ax.add_patch(arrow)
            if show_edge_labels:
                mx, my = (coords[i] + coords[j]) / 2
                ax.text(mx, my, f"{w:.2f}", ha="center", va="center", fontsize=8)

    ax.set_axis_off()
    ax.set_aspect("equal")
    return ax


def heatmap_single(
    matrix: np.ndarray,
    *,
    sender_names: Optional[Iterable[str]] = None,
    receiver_names: Optional[Iterable[str]] = None,
    group: Optional[Mapping[str, str]] = None,
    ligand_abundance: Optional[np.ndarray] = None,
    target_abundance: Optional[np.ndarray] = None,
    cmap: str = "bwr",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw a heatmap with optional group annotations and abundance bars."""

    matrix = np.asarray(matrix)
    sender_names = list(sender_names or range(matrix.shape[0]))
    receiver_names = list(receiver_names or range(matrix.shape[1]))

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5])
        ax_top = fig.add_subplot(gs[0, 0]) if target_abundance is not None else None
        ax_right = fig.add_subplot(gs[1, 1]) if ligand_abundance is not None else None
        ax = fig.add_subplot(gs[1, 0])
    else:
        ax_top = ax_right = None

    sns.heatmap(
        matrix,
        xticklabels=receiver_names,
        yticklabels=sender_names,
        cmap=cmap,
        cbar_kws={"label": "Communication"},
        ax=ax,
    )

    if group is not None:
        groups = [group.get(n, n) for n in sender_names]
        unique_groups = list(dict.fromkeys(groups))
        colors = sc_palette(len(unique_groups))
        color_map = {g: c for g, c in zip(unique_groups, colors)}
        for tick, grp in zip(ax.get_yticklabels(), groups):
            tick.set_color(color_map[grp])

        groups_col = [group.get(n, n) for n in receiver_names]
        unique_groups_col = list(dict.fromkeys(groups_col))
        colors_col = sc_palette(len(unique_groups_col))
        color_map_col = {g: c for g, c in zip(unique_groups_col, colors_col)}
        for tick, grp in zip(ax.get_xticklabels(), groups_col):
            tick.set_color(color_map_col[grp])

    if ax_top is not None and target_abundance is not None:
        sns.barplot(x=receiver_names, y=target_abundance, ax=ax_top, color="skyblue")
        ax_top.set_ylabel("Target abundance")
        ax_top.set_xlabel("")
        ax_top.tick_params(axis="x", labelrotation=90)

    if ax_right is not None and ligand_abundance is not None:
        sns.barplot(y=sender_names, x=ligand_abundance, ax=ax_right, color="skyblue")
        ax_right.set_xlabel("Ligand abundance")
        ax_right.set_ylabel("")

    return ax

