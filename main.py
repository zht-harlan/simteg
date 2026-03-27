import gc
import csv
import json
import logging
import os
import os.path as osp
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.args import LINK_PRED_DATASETS, load_args, parse_args, save_args
from src.run import train
from src.utils import dict_append, is_dist, set_logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def set_single_env(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    torch.cuda.empty_cache()
    if is_dist():
        dist.destroy_process_group()
    gc.collect()


def set_seed(random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if is_dist():
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    if is_dist():
        gpus = ",".join([str(_) for _ in range(int(os.environ["WORLD_SIZE"]))])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def print_metrics(metrics: dict, metric_type):
    results = {key: (np.mean(value), np.std(value)) for key, value in metrics.items()}
    logger.critical(
        f"{metric_type} Metrics:\n"
        + "\n".join(
            "{}: {} +/- {} ".format(k, _mean, _std)
            for k, (_mean, _std) in results.items()
        )
    )


def summarize_metrics(metrics):
    return {
        key: {
            "mean": float(np.mean(value)),
            "std": float(np.std(value)),
            "values": [float(v) for v in value],
        }
        for key, value in metrics.items()
    }


def save_results(args, seeds, val_metrics, test_metrics):
    if int(os.getenv("RANK", -1)) > 0:
        return

    results = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "suffix": args.suffix,
        "n_exps": args.n_exps,
        "start_seed": args.start_seed,
        "seeds": seeds,
        "val": summarize_metrics(val_metrics),
        "test": summarize_metrics(test_metrics),
        "runs": [],
    }

    for idx, seed in enumerate(seeds):
        run_result = {
            "run_idx": idx,
            "seed": int(seed),
            "val": {key: float(val_metrics[key][idx]) for key in val_metrics},
            "test": {key: float(test_metrics[key][idx]) for key in test_metrics},
        }
        results["runs"].append(run_result)

    results_path = osp.join(args.output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_csv_path = osp.join(".", "results_summary.csv")
    fieldnames = [
        "dataset",
        "model_type",
        "suffix",
        "n_exps",
        "start_seed",
        "seeds",
        "val_acc_mean",
        "val_acc_std",
        "val_macro_f1_mean",
        "val_macro_f1_std",
        "test_acc_mean",
        "test_acc_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "results_json",
    ]
    row = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "suffix": args.suffix,
        "n_exps": args.n_exps,
        "start_seed": args.start_seed,
        "seeds": " ".join(str(seed) for seed in seeds),
        "val_acc_mean": results["val"].get("acc", {}).get("mean"),
        "val_acc_std": results["val"].get("acc", {}).get("std"),
        "val_macro_f1_mean": results["val"].get("macro_f1", {}).get("mean"),
        "val_macro_f1_std": results["val"].get("macro_f1", {}).get("std"),
        "test_acc_mean": results["test"].get("acc", {}).get("mean"),
        "test_acc_std": results["test"].get("acc", {}).get("std"),
        "test_macro_f1_mean": results["test"].get("macro_f1", {}).get("mean"),
        "test_macro_f1_std": results["test"].get("macro_f1", {}).get("std"),
        "results_json": results_path,
    }

    existing_rows = []
    if osp.exists(summary_csv_path):
        with open(summary_csv_path, "r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))

    existing_rows = [
        existing
        for existing in existing_rows
        if not (
            existing["dataset"] == args.dataset
            and existing["model_type"] == args.model_type
            and existing["suffix"] == args.suffix
        )
    ]
    existing_rows.append(row)

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    logger.critical(f"saved structured results to {results_path}")
    logger.critical(f"updated summary csv at {summary_csv_path}")


def main(args):
    set_logging()
    if is_dist():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        set_single_env(rank, world_size)

    save_args(args, args.output_dir)

    if args.dataset in LINK_PRED_DATASETS:
        val_metrics_list = {"mrr": [], "hits@1": [], "hits@3": [], "hits@10": []}
        test_metrics_list = {"mrr": [], "hits@1": [], "hits@3": [], "hits@10": []}
    else:
        val_metrics_list = {"acc": [], "macro_f1": []}
        test_metrics_list = {"acc": [], "macro_f1": []}
    seeds = []

    for i, random_seed in enumerate(range(args.n_exps)):
        random_seed += args.start_seed
        set_seed(random_seed)
        logger.critical(f"{i}-th run with seed {random_seed}")
        args.random_seed = random_seed
        seeds.append(random_seed)
        logger.info(args)

        test_metrics, val_metrics = train(args, return_value="test")

        val_metrics_list = dict_append(val_metrics_list, val_metrics)
        test_metrics_list = dict_append(test_metrics_list, test_metrics)

        print_metrics(val_metrics_list, "Current Val")
        print_metrics(test_metrics_list, "Current Test")

    cleanup()
    print_metrics(val_metrics_list, "Final Val")
    print_metrics(test_metrics_list, "Final Test")
    if args.dataset not in LINK_PRED_DATASETS:
        save_results(args, seeds, val_metrics_list, test_metrics_list)


if __name__ == "__main__":
    args = parse_args()
    save_args(args, args.output_dir)
    main(args)
