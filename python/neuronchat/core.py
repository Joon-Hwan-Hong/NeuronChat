from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError:  # pragma: no cover - optional dependency
    ad = None


@dataclass
class NeuronChat:
    """A minimal Python version of the NeuronChat container."""

    data_raw: Optional[np.ndarray] = None
    data: Optional[np.ndarray] = None
    data_signaling: pd.DataFrame = field(default_factory=pd.DataFrame)
    net0: List[np.ndarray] = field(default_factory=list)
    pvalue: List[np.ndarray] = field(default_factory=list)
    net: List[np.ndarray] = field(default_factory=list)
    net_analysis: Dict[str, Any] = field(default_factory=dict)
    fc: np.ndarray = field(default_factory=lambda: np.array([]))
    info: np.ndarray = field(default_factory=lambda: np.array([]))
    ligand_abundance: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    target_abundance: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    idents: pd.Categorical = field(default_factory=lambda: pd.Categorical([]))
    DB: Dict[str, Any] = field(default_factory=dict)
    LR: List[str] = field(default_factory=list)
    dr: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_anndata(cls, adata: "ad.AnnData", group_by: str) -> "NeuronChat":
        """Create :class:`NeuronChat` from :class:`anndata.AnnData`."""
        if ad is None:
            raise ImportError("anndata is required to use from_anndata")

        if group_by not in adata.obs:
            raise ValueError(f"{group_by} not found in adata.obs")

        data = adata.X.T  # genes x cells
        meta = adata.obs.copy()
        idents = pd.Categorical(meta[group_by])

        obj = cls(
            data_raw=data,
            data=data,
            meta=meta,
            idents=idents,
        )
        obj.options["mode"] = "single"
        return obj

    # ------------------------------------------------------------------
    # Placeholder methods translated from the R package
    def run_NeuronChat(
        self,
        sender: Optional[List[str]] = None,
        receiver: Optional[List[str]] = None,
        M: int = 100,
        fdr: float = 0.05,
    ) -> None:
        """Calculate communication strength matrices.

        This implementation is a minimal placeholder port of the R
        ``run_NeuronChat`` function using ``numpy`` and ``pandas``.
        """

        if self.data is None:
            raise ValueError("data is missing")

        genes, cells = self.data.shape
        self.net0 = [np.zeros((genes, genes))]
        self.pvalue = [np.ones((genes, genes))]
        self.net = [np.zeros((genes, genes))]
        self.net_analysis["method"] = "placeholder"

    def net_aggregation(self) -> None:
        """Aggregate network strengths. Placeholder implementation."""
        if not self.net:
            return
        self.net_analysis["aggregate"] = sum(np.sum(n) for n in self.net)

    def computeNetSimilarityPairwise_Neuron(self) -> np.ndarray:
        """Return pairwise similarity across nets. Placeholder."""
        if not self.net:
            return np.empty((0, 0))
        n = len(self.net)
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i <= j:
                    s = np.corrcoef(self.net[i].ravel(), self.net[j].ravel())[0, 1]
                    sim[i, j] = sim[j, i] = s
        return sim
