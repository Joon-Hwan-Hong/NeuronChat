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

    def net_aggregation(
        self,
        method: str = "weight",
        cut_off: float = 0.05,
    ) -> np.ndarray:
        """Aggregate communication networks using :func:`net_aggregation`."""

        if not self.net:
            return np.empty((0, 0))

        net_agg = net_aggregation(self.net, method=method, cut_off=cut_off)
        self.net_analysis["aggregate"] = net_agg
        return net_agg

    # ------------------------------------------------------------------
    def computeNetSimilarityPairwise_Neuron(
        self,
        slot_name: str = "net",
        type: str = "functional",
        comparison: Optional[List[int]] = None,
        k: Optional[int] = None,
        thresh: Optional[float] = None,
    ) -> np.ndarray:
        """Compute network similarity using :func:`computeNetSimilarityPairwise_Neuron`."""

        sim = computeNetSimilarityPairwise_Neuron(
            self,
            slot_name=slot_name,
            type=type,
            comparison=comparison,
            k=k,
            thresh=thresh,
        )
        return sim


def net_aggregation(
    net_list: List[np.ndarray],
    method: str = "weight",
    cut_off: float = 0.05,
) -> np.ndarray:
    """Aggregate a list of communication networks."""

    if len(net_list) == 0:
        return np.empty((0, 0))

    method = method.lower()
    net_agg = np.zeros_like(net_list[0], dtype=float)

    for mat in net_list:
        mat = np.asarray(mat, dtype=float)
        if method == "weighted_count":
            net_agg += mat.sum() * (mat > 0)
        elif method == "count":
            net_agg += (mat > 0).astype(float)
        elif method == "weighted_count2":
            thresh = np.quantile(mat, cut_off)
            denom = 1e-6 + (mat > thresh).sum()
            net_agg += mat.sum() / denom * (mat > thresh)
        elif method == "weight_threshold":
            thresh = np.quantile(mat, cut_off)
            net_agg += mat * (mat > thresh)
        else:  # "weight"
            net_agg += mat

    return net_agg


def computeNetSimilarityPairwise_Neuron(
    obj: "NeuronChat",
    slot_name: str = "net",
    type: str = "functional",
    comparison: Optional[List[int]] = None,
    k: Optional[int] = None,
    thresh: Optional[float] = None,
) -> np.ndarray:
    """Compute signaling network similarity for any pair of datasets."""

    nets = getattr(obj, slot_name, None)
    if nets is None or len(nets) == 0:
        return np.empty((0, 0))

    if comparison is None:
        if "datasets" in obj.meta:
            comparison = list(range(len(obj.meta["datasets"].unique())))
        else:
            comparison = list(range(len(nets)))

    net_arrays: List[np.ndarray] = []
    for idx in comparison:
        arr = np.stack(nets[idx], axis=2) if isinstance(nets[idx], list) else np.asarray(nets[idx])
        net_arrays.append(arr)

    net_dims = [arr.shape[2] for arr in net_arrays]
    nnet = sum(net_dims)
    pos = np.cumsum([0] + net_dims)

    if k is None:
        k = int(np.ceil(np.sqrt(nnet))) + (1 if nnet > 25 else 0)

    if thresh is not None:
        for i, arr in enumerate(net_arrays):
            nz = arr[arr != 0]
            if nz.size > 0:
                q = np.quantile(nz, thresh)
                arr[arr < q] = 0
            net_arrays[i] = arr

    S_signalings = np.zeros((nnet, nnet), dtype=float)

    def _get_slice(index: int) -> np.ndarray:
        ds_idx = next(j for j, p in enumerate(pos[1:], start=1) if index < p) - 1
        offset = index - pos[ds_idx]
        return net_arrays[ds_idx][:, :, offset]

    type = type.lower()
    for i in range(nnet):
        Gi = (_get_slice(i) > 0).astype(float)
        for j in range(nnet):
            Gj = (_get_slice(j) > 0).astype(float)
            inter = np.logical_and(Gi, Gj).sum()
            union = np.logical_or(Gi, Gj).sum()
            if type == "structural":
                val = inter / union if union > 0 else 0.0
                S_signalings[i, j] = val
            else:
                S_signalings[i, j] = inter / union if union > 0 else 0.0

    S_signalings[np.isnan(S_signalings)] = 0
    np.fill_diagonal(S_signalings, 1)

    SNN = _build_snn(S_signalings, k=k)
    Similarity = S_signalings * SNN

    obj.net_analysis.setdefault("similarity", {}).setdefault(type, {})["matrix"] = Similarity
    return Similarity


def _build_snn(sim: np.ndarray, k: int, prune: float = 1 / 15) -> np.ndarray:
    """Construct a simple shared nearest neighbor matrix."""

    n = sim.shape[0]
    ranks = np.argsort(-sim, axis=1)
    neigh = ranks[:, 1 : k + 1]
    snn = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in neigh[i]:
            shared = np.intersect1d(neigh[i], neigh[j]).size
            weight = shared / k
            if weight >= prune:
                snn[i, j] = snn[j, i] = weight
    np.fill_diagonal(snn, 1)
    return snn
