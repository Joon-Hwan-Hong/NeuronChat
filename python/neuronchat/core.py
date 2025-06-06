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
        K: float = 0.5,
        method: Optional[str] = None,
        mean_method: Optional[str] = None,
        strict: int = 1,
    ) -> None:
        """Calculate communication strength matrices with permutation test."""

        if self.data_signaling.empty:
            raise ValueError("data_signaling is missing")

        interactionDB = self.DB
        net0_all: List[np.ndarray] = []
        net_all: List[np.ndarray] = []
        pvalue_all: List[np.ndarray] = []
        FC_all = np.zeros(len(interactionDB))
        info_all = np.zeros(len(interactionDB))

        net_rownames = sorted(self.data_signaling["cell_subclass"].unique())
        if sender is None:
            sender = net_rownames
        if receiver is None:
            receiver = net_rownames

        ligand_abundance_all = np.zeros((len(net_rownames), len(interactionDB)))
        target_abundance_all = np.zeros_like(ligand_abundance_all)

        for j, (interaction_name, inter) in enumerate(interactionDB.items()):
            lig_contributor = inter["lig_contributor"]
            receptor_subunit = inter["receptor_subunit"]
            lig_contributor_group = inter["lig_contributor_group"]
            lig_contributor_coeff = inter["lig_contributor_coeff"]
            receptor_subunit_group = inter["receptor_subunit_group"]
            receptor_subunit_coeff = inter["receptor_subunit_coeff"]

            lig_boolean_group = [g for g, gene in zip(lig_contributor_group, lig_contributor) if gene in self.data_signaling.columns]
            rec_boolean_group = [g for g, gene in zip(receptor_subunit_group, receptor_subunit) if gene in self.data_signaling.columns]

            if strict == 1:
                lig_boolean = np.prod([grp in lig_boolean_group for grp in set(lig_contributor_group)])
                rec_boolean = np.prod([grp in rec_boolean_group for grp in set(receptor_subunit_group)])
            else:
                lig_boolean = np.sum([grp in lig_boolean_group for grp in set(lig_contributor_group)]) > 0
                rec_boolean = np.sum([grp in rec_boolean_group for grp in set(receptor_subunit_group)]) > 0

            if lig_boolean * rec_boolean == 0:
                net_tmp = np.zeros((len(sender), len(receiver)))
                pval_tmp = np.zeros_like(net_tmp)
                net0_tmp = np.zeros_like(net_tmp)
                fc_tmp = 0.0
                info_tmp = 0.0
                lig_score = np.zeros(len(net_rownames))
                rec_score = np.zeros(len(net_rownames))
            else:
                tmp = neuron_chat_downstream(
                    self.data_signaling.copy(),
                    sender,
                    receiver,
                    interaction_name,
                    lig_contributor,
                    lig_contributor_group,
                    lig_contributor_coeff,
                    receptor_subunit,
                    receptor_subunit_group,
                    receptor_subunit_coeff,
                    [],
                    [],
                    {},
                    0,
                    M,
                    fdr,
                    K,
                    method,
                    mean_method,
                )
                net_tmp = tmp["net"]
                net0_tmp = tmp["net0"]
                pval_tmp = tmp["pvalue"]
                fc_tmp = tmp["FC"]
                info_tmp = tmp["info"]
                lig_score = tmp["ligand.abundance"]
                rec_score = tmp["target.abundance"]

            net_all.append(net_tmp)
            net0_all.append(net0_tmp)
            pvalue_all.append(pval_tmp)
            FC_all[j] = fc_tmp
            info_all[j] = info_tmp
            ligand_abundance_all[:, j] = lig_score
            target_abundance_all[:, j] = rec_score

        self.net0 = net0_all
        self.pvalue = pvalue_all
        self.net = net_all
        self.fc = FC_all
        self.info = info_all
        self.ligand_abundance = ligand_abundance_all
        self.target_abundance = target_abundance_all

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


def _cal_expr_by_group(df: pd.DataFrame, gene_used: List[str], mean_method: Optional[str] = None) -> pd.DataFrame:
    """Aggregate expression by cell group."""
    df_used = df[gene_used + ["cell_subclass"]]
    if mean_method == "mean":
        yy = df_used.groupby("cell_subclass")[gene_used].mean()
    else:
        q1 = df_used.groupby("cell_subclass")[gene_used].quantile(0.25)
        q2 = df_used.groupby("cell_subclass")[gene_used].quantile(0.50)
        q3 = df_used.groupby("cell_subclass")[gene_used].quantile(0.75)
        yy = 0.25 * q1 + 0.5 * q2 + 0.25 * q3
    return yy


def cal_prob_mtx_downstream(
    df: pd.DataFrame,
    sender: List[str],
    receiver: List[str],
    lig_contributor_new: List[str],
    lig_contributor_group: List[int],
    lig_contributor_coeff: List[int],
    receptor_subunit_new: List[str],
    receptor_subunit_group: List[int],
    receptor_subunit_coeff: List[int],
    targets_up: List[str],
    targets_down: List[str],
    targets_nichenet: Dict[str, float],
    N: int,
    K: float = 0.5,
    method: Optional[str] = None,
    mean_method: Optional[str] = None,
) -> Dict[str, Any]:
    gene_used = lig_contributor_new + receptor_subunit_new
    expr_gene = _cal_expr_by_group(df, gene_used, mean_method)
    cell_names = list(expr_gene.index)
    cellgroup_number = expr_gene.shape[0]

    if method is None:
        # ligand
        lig_tmp = np.zeros((cellgroup_number, len(lig_contributor_coeff)))
        for i in range(1, len(lig_contributor_coeff) + 1):
            ind_i = [k for k, g in enumerate(lig_contributor_group) if g == i]
            if len(ind_i) == 1:
                lig_tmp[:, i - 1] = expr_gene.iloc[:, ind_i].to_numpy().ravel()
            elif len(ind_i) > 1:
                lig_tmp[:, i - 1] = expr_gene.iloc[:, ind_i].mean(axis=1)
            else:
                lig_tmp[:, i - 1] = 1
        if len(lig_contributor_coeff) > 1:
            rep_coeff = []
            for g, c in enumerate(lig_contributor_coeff, start=1):
                rep_coeff.extend([g - 1] * c)
            lig_contributor_expr = np.exp(np.log(lig_tmp[:, rep_coeff]).mean(axis=1))
        else:
            lig_contributor_expr = lig_tmp[:, 0]

        # receptor
        rec_tmp = np.zeros((cellgroup_number, len(receptor_subunit_coeff)))
        for i in range(1, len(receptor_subunit_coeff) + 1):
            ind_i = [k for k, g in enumerate(receptor_subunit_group) if g == i]
            offset = len(lig_contributor_new)
            if len(ind_i) == 1:
                rec_tmp[:, i - 1] = expr_gene.iloc[:, offset + np.array(ind_i)].to_numpy().ravel()
            elif len(ind_i) > 1:
                rec_tmp[:, i - 1] = expr_gene.iloc[:, offset + np.array(ind_i)].mean(axis=1)
            else:
                rec_tmp[:, i - 1] = 1
        if len(receptor_subunit_coeff) > 1:
            rep_coeff = []
            for g, c in enumerate(receptor_subunit_coeff, start=1):
                rep_coeff.extend([g - 1] * c)
            receptor_subunit_expr = np.exp(np.log(rec_tmp[:, rep_coeff]).mean(axis=1))
        else:
            receptor_subunit_expr = rec_tmp[:, 0]

        prob_mtx0 = np.outer(lig_contributor_expr, receptor_subunit_expr)
        prob_mtx = prob_mtx0

    elif method == "CellChat":
        if len(lig_contributor_new) > 1:
            lig_contributor_expr = np.exp(np.log(expr_gene.iloc[:, : len(lig_contributor_new)]).mean(axis=1))
        else:
            lig_contributor_expr = expr_gene.iloc[:, 0]
        if len(receptor_subunit_new) > 1:
            receptor_subunit_expr = np.exp(
                np.log(expr_gene.iloc[:, len(lig_contributor_new) :]).mean(axis=1)
            )
        else:
            receptor_subunit_expr = expr_gene.iloc[:, len(lig_contributor_new)]

        hill = lambda x, k: x**2 / (x**2 + k**2)
        prob_mtx0 = np.outer(hill(lig_contributor_expr, K), hill(receptor_subunit_expr, K))
        prob_mtx = prob_mtx0

    elif method == "CellPhoneDB":
        if len(lig_contributor_new) > 1:
            lig_contributor_expr = expr_gene.iloc[:, : len(lig_contributor_new)].min(axis=1)
        else:
            lig_contributor_expr = expr_gene.iloc[:, 0]
        if len(receptor_subunit_new) > 1:
            receptor_subunit_expr = expr_gene.iloc[:, len(lig_contributor_new) :].min(axis=1)
        else:
            receptor_subunit_expr = expr_gene.iloc[:, len(lig_contributor_new)]

        prob_mtx0_lig = np.outer(lig_contributor_expr, np.ones(len(cell_names)))
        prob_mtx0_rec = np.outer(np.ones(len(cell_names)), receptor_subunit_expr)
        prob_mtx0 = (prob_mtx0_lig + prob_mtx0_rec) / 2
        prob_mtx0 = prob_mtx0 * (prob_mtx0_lig > 0) * (prob_mtx0_rec > 0)
        prob_mtx = prob_mtx0
    else:
        raise ValueError("Unknown method")

    prob_mtx_df = pd.DataFrame(prob_mtx, index=cell_names, columns=cell_names)
    prob_mtx0_df = pd.DataFrame(prob_mtx0, index=cell_names, columns=cell_names)
    prob_mtx = prob_mtx_df.loc[sender, receiver].to_numpy()
    prob_mtx0 = prob_mtx0_df.loc[sender, receiver].to_numpy()

    return {
        "prob_mtx": prob_mtx,
        "ligand_score": np.array(lig_contributor_expr),
        "receptor_score": np.array(receptor_subunit_expr),
        "cell_group": cell_names,
        "prob_mtx0": prob_mtx0,
        "targets_up_score": np.zeros(cellgroup_number),
        "targets_down_score": np.zeros(cellgroup_number),
        "targets_nichenet_score": np.zeros(cellgroup_number),
    }


def neuron_chat_downstream(
    df: pd.DataFrame,
    sender: List[str],
    receiver: List[str],
    interaction_name: str,
    lig_contributor: List[str],
    lig_contributor_group: List[int],
    lig_contributor_coeff: List[int],
    receptor_subunit: List[str],
    receptor_subunit_group: List[int],
    receptor_subunit_coeff: List[int],
    targets_up: List[str],
    targets_down: List[str],
    targets_nichenet: Dict[str, float],
    N: int,
    M: int,
    fdr: float,
    K: float = 0.5,
    method: Optional[str] = None,
    mean_method: Optional[str] = None,
) -> Dict[str, Any]:
    ind_lig = [i for i, g in enumerate(lig_contributor) if g in df.columns]
    lig_contributor_new = [lig_contributor[i] for i in ind_lig]
    ind_re = [i for i, g in enumerate(receptor_subunit) if g in df.columns]
    receptor_subunit_new = [receptor_subunit[i] for i in ind_re]
    lig_contributor_group = [lig_contributor_group[i] for i in ind_lig]
    receptor_subunit_group = [receptor_subunit_group[i] for i in ind_re]
    targets_up = [t for t in targets_up if t in df.columns]
    targets_down = [t for t in targets_down if t in df.columns]
    targets_nichenet = {k: v for k, v in targets_nichenet.items() if k in df.columns}

    my_list = cal_prob_mtx_downstream(
        df,
        sender,
        receiver,
        lig_contributor_new,
        lig_contributor_group,
        lig_contributor_coeff,
        receptor_subunit_new,
        receptor_subunit_group,
        receptor_subunit_coeff,
        targets_up,
        targets_down,
        targets_nichenet,
        N,
        K,
        method,
        mean_method,
    )

    prob_mtx = my_list["prob_mtx"]
    FC_lig = prob_mtx.max() - prob_mtx.min() if prob_mtx.size else 0
    FC_rec = FC_lig
    FC = max(FC_lig, FC_rec)

    if M == 0 or prob_mtx.max() == 0:
        net = prob_mtx
        pvalue = np.full_like(net, np.nan)
    else:
        prob_perm = np.zeros((prob_mtx.shape[0], prob_mtx.shape[1], M))
        for j in range(M):
            df_j = df.copy()
            mask_sender = df_j["cell_subclass"].isin(sender)
            df_j.loc[mask_sender, "cell_subclass"] = np.random.permutation(df_j.loc[mask_sender, "cell_subclass"])
            mask_receiver = df_j["cell_subclass"].isin(receiver)
            df_j.loc[mask_receiver, "cell_subclass"] = np.random.permutation(df_j.loc[mask_receiver, "cell_subclass"])

            tmp = cal_prob_mtx_downstream(
                df_j,
                sender,
                receiver,
                lig_contributor_new,
                lig_contributor_group,
                lig_contributor_coeff,
                receptor_subunit_new,
                receptor_subunit_group,
                receptor_subunit_coeff,
                targets_up,
                targets_down,
                targets_nichenet,
                N,
                K,
                method,
                mean_method,
            )
            prob_perm[:, :, j] = (tmp["prob_mtx"] > prob_mtx).astype(float)

        pvalue = prob_perm.sum(axis=2) / M
        pval_vec = pvalue.ravel()
        order = np.argsort(pval_vec)
        alpha = fdr * (np.arange(1, len(pval_vec) + 1)) / len(pval_vec)
        k_idx = order[pval_vec[order] < alpha]
        net = np.zeros_like(pvalue)
        flat_prob = prob_mtx.ravel()
        net.ravel()[k_idx] = flat_prob[k_idx]

    return {
        "net": net,
        "FC": FC,
        "pvalue": pvalue,
        "net0": my_list["prob_mtx"],
        "info": float(net.sum()),
        "ligand.abundance": my_list["ligand_score"],
        "target.abundance": my_list["receptor_score"],
    }
