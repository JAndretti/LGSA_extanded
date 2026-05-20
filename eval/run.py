"""LG-SA evaluation entry point.

Usage:
  uv run python -m eval.run --dataset random --project <P> --group <G> \
      --dim 100 --max_load 50 [other flags]

The --dataset flag dispatches to handler_<dataset>.py inside this folder.
Only handler_random is implemented; the other handlers are stubs that raise
NotImplementedError.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

HANDLERS: dict[str, str] = {
    "random":  "eval.handler_random",
    "X":       "eval.handler_X",
    "X_batch": "eval.handler_X_batch",
    "XL":      "eval.handler_XL",
    "XML":     "eval.handler_XML",
}


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def main() -> None:
    # Pass 1: extract --dataset so we can import the right handler.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", choices=HANDLERS.keys(), default=None)
    pre_args, _ = pre.parse_known_args()
    handler = (
        importlib.import_module(HANDLERS[pre_args.dataset])
        if pre_args.dataset else None
    )

    # Pass 2: full parser, including handler-specific args.
    parser = argparse.ArgumentParser(description="LG-SA evaluation entry point")
    parser.add_argument("--dataset", required=True, choices=HANDLERS.keys())
    parser.add_argument("--project", required=True,
                        help="W&B project name (dir under wandb/)")
    parser.add_argument("--group", required=True,
                        help="W&B group name (dir under wandb/<project>/)")
    parser.add_argument(
        "--INIT", type=str, default="random",
        choices=["random", "isolate", "sweep", "nearest_neighbor",
                 "Clark_and_Wright", "farthest_insertion"],
    )
    parser.add_argument("--OUTER_STEPS", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=_auto_device())
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "bfloat16", "float16"],
    )
    if handler is not None:
        handler.add_args(parser)
    args = parser.parse_args()
    args.torch_dtype = _DTYPE_MAP[args.dtype]

    if handler is None:
        parser.error("argument --dataset is required")
    handler.run(args)


if __name__ == "__main__":
    main()
