import ast
import os.path as osp

import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import Amazon, Planetoid
from torch_geometric.data import Data

from ..utils import SimpleNodeEvaluator


def _mask_to_index(mask):
    return mask.nonzero(as_tuple=False).view(-1)


def _balanced_split(y, num_train_per_class=20, num_val_per_class=30, seed=42):
    y = y.view(-1)
    generator = torch.Generator().manual_seed(seed)
    train_idx, valid_idx, test_idx = [], [], []
    num_classes = int(y.max().item()) + 1
    for cls in range(num_classes):
        cls_idx = (y == cls).nonzero(as_tuple=False).view(-1)
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=generator)]
        train_end = min(num_train_per_class, perm.numel())
        val_end = min(train_end + num_val_per_class, perm.numel())
        train_idx.append(perm[:train_end])
        valid_idx.append(perm[train_end:val_end])
        test_idx.append(perm[val_end:])
    return {
        "train": torch.cat(train_idx, dim=0),
        "valid": torch.cat(valid_idx, dim=0),
        "test": torch.cat(test_idx, dim=0),
    }


def _stratified_split(y, train_ratio=0.2, valid_ratio=0.2, seed=42):
    y = y.view(-1)
    generator = torch.Generator().manual_seed(seed)
    train_idx, valid_idx, test_idx = [], [], []
    num_classes = int(y.max().item()) + 1
    for cls in range(num_classes):
        cls_idx = (y == cls).nonzero(as_tuple=False).view(-1)
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=generator)]
        train_end = max(1, int(perm.numel() * train_ratio))
        valid_end = train_end + max(1, int(perm.numel() * valid_ratio))
        if valid_end >= perm.numel():
            train_end = max(1, perm.numel() - 2)
            valid_end = train_end + 1
        train_idx.append(perm[:train_end])
        valid_idx.append(perm[train_end:valid_end])
        test_idx.append(perm[valid_end:])
    return {
        "train": torch.cat(train_idx, dim=0),
        "valid": torch.cat(valid_idx, dim=0),
        "test": torch.cat(test_idx, dim=0),
    }


def _title_case_dataset_name(name):
    return {
        "children": "Children",
        "history": "History",
        "photo": "Photo",
    }.get(name, name)


def _find_existing_path(candidates):
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return None


def _resolve_custom_dataset_paths(root, name):
    title_name = _title_case_dataset_name(name)
    csv_path = _find_existing_path(
        [
            osp.join(root, "CSTAG", title_name, f"{title_name}.csv"),
            osp.join(root, "CSTAG", name, f"{name}.csv"),
            osp.join(root, title_name, f"{title_name}.csv"),
            osp.join(root, name, f"{name}.csv"),
        ]
    )
    if csv_path is None:
        raise FileNotFoundError(f"Unable to locate csv for dataset '{name}' under {root}.")
    return osp.dirname(csv_path), csv_path


def _resolve_feature_path(root, dataset_dir, name):
    title_name = _title_case_dataset_name(name)
    candidates = [
        osp.join(root, "manual_features", f"{name}_plm.npy"),
        osp.join(root, "manual_features", f"{title_name.lower()}_plm.npy"),
        osp.join(root, "manual_features", f"{title_name}_plm.npy"),
        osp.join(dataset_dir, "Feature", f"{title_name}_roberta_base_512_cls.npy"),
        osp.join(dataset_dir, "Feature", f"{name}_roberta_base_512_cls.npy"),
        osp.join(dataset_dir, "Feature", f"{title_name}_Qwen2.5_7B_256_mean.npy"),
        osp.join(dataset_dir, "Feature", f"{name}_Qwen2.5_7B_256_mean.npy"),
        osp.join(dataset_dir, "Feature", f"{title_name.lower()}_Qwen2.5_7B_256_mean.npy"),
        osp.join(dataset_dir, "Feature", f"{title_name}_qwen2.5_7b_256_mean.npy"),
        osp.join(dataset_dir, "Feature", f"{name}_qwen2.5_7b_256_mean.npy"),
    ]
    feature_path = _find_existing_path(candidates)
    if feature_path is None:
        raise FileNotFoundError(
            f"Unable to locate PLM feature file for dataset '{name}'. "
            f"Searched: {', '.join(candidates)}"
        )
    return feature_path


def _parse_neighbor_list(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [int(item) for item in parsed]
    raise ValueError(f"Unsupported neighbour field: {value!r}")


def _build_edge_index_from_dataframe(df):
    if "node_id" not in df.columns or "neighbour" not in df.columns:
        raise ValueError("Custom dataset CSV must contain 'node_id' and 'neighbour' columns.")
    sources = []
    targets = []
    for row in df.itertuples(index=False):
        node_id = int(getattr(row, "node_id"))
        neighbours = _parse_neighbor_list(getattr(row, "neighbour"))
        for neighbour in neighbours:
            sources.append(node_id)
            targets.append(neighbour)
    if not sources:
        raise ValueError("No edges were parsed from the 'neighbour' column.")
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return torch.cat([edge_index, edge_index[[1, 0], :]], dim=1).contiguous()


def _coerce_edge_index(value):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if isinstance(value, dict):
        for key in ["edge_index", "edges", "graph"]:
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        raise TypeError("Edge file must resolve to a tensor-like value.")
    if value.dim() != 2:
        raise ValueError(f"edge_index must be 2D, got shape {tuple(value.shape)}.")
    if value.size(0) != 2 and value.size(1) == 2:
        value = value.t()
    if value.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], got {tuple(value.shape)}.")
    return value.long().contiguous()


def _load_edge_index(dataset_dir, df):
    edge_path = _find_existing_path(
        [
            osp.join(dataset_dir, "edge_index.pt"),
            osp.join(dataset_dir, "edge_index.npy"),
            osp.join(dataset_dir, "edges.pt"),
            osp.join(dataset_dir, "edges.npy"),
            osp.join(dataset_dir, "graph.pt"),
            osp.join(dataset_dir, "graph.npy"),
            osp.join(dataset_dir, "adjacency.pt"),
            osp.join(dataset_dir, "adjacency.npy"),
        ]
    )
    if edge_path is None:
        return _build_edge_index_from_dataframe(df)
    if edge_path.endswith((".pt", ".pth")):
        loaded = torch.load(edge_path, map_location="cpu", weights_only=False)
    else:
        loaded = np.load(edge_path, allow_pickle=True)
    return _coerce_edge_index(loaded)


def _load_custom_node_dataset(name, root="data", seed=42):
    dataset_dir, csv_path = _resolve_custom_dataset_paths(root, name)
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"Dataset csv is missing required 'label' column: {csv_path}")
    if "node_id" in df.columns:
        df = df.sort_values("node_id").reset_index(drop=True)

    feature_path = _resolve_feature_path(root, dataset_dir, name)
    x = torch.from_numpy(np.load(feature_path).astype(np.float32))
    y = torch.from_numpy(df["label"].to_numpy()).long()
    if x.size(0) != y.size(0):
        raise ValueError(
            f"Feature rows ({x.size(0)}) do not match label rows ({y.size(0)}) for dataset '{name}'."
        )
    edge_index = _load_edge_index(dataset_dir, df)
    data = Data(x=x, edge_index=edge_index, y=y)
    split_idx = _stratified_split(y, train_ratio=0.2, valid_ratio=0.2, seed=seed)
    return data, split_idx, SimpleNodeEvaluator()


def load_pyg_node_dataset(name, root="data", seed=42):
    if name in ["cora", "pubmed"]:
        dataset_name = {"cora": "Cora", "pubmed": "PubMed"}[name]
        dataset = Planetoid(root=osp.join(root, name), name=dataset_name)
        data = dataset[0]
        split_idx = {
            "train": _mask_to_index(data.train_mask),
            "valid": _mask_to_index(data.val_mask),
            "test": _mask_to_index(data.test_mask),
        }
    elif name == "amazon-photo":
        dataset = Amazon(root=osp.join(root, "amazon-photo"), name="Photo")
        data = dataset[0]
        split_idx = _balanced_split(data.y, seed=seed)
    elif name in ["children", "history", "photo"]:
        return _load_custom_node_dataset(name, root=root, seed=seed)
    else:
        raise NotImplementedError(f"Unsupported dataset: {name}")

    return data, split_idx, SimpleNodeEvaluator()
