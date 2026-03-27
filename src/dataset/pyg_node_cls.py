import os.path as osp

import torch
from torch_geometric.datasets import Amazon, Planetoid

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
    else:
        raise NotImplementedError(f"Unsupported dataset: {name}")

    return data, split_idx, SimpleNodeEvaluator()
