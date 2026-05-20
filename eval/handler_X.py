"""CVRPLib Set X handler — not implemented in this iteration.

See `eval/handler_random.py` for the reference structure. Add the .vrp loader
in `eval_io.py` and the rounded-cost convention in `eval/costs.py` when
implementing.
"""
import argparse


def add_args(parser: argparse.ArgumentParser) -> None:
    raise NotImplementedError("handler_X is a stub — see handler_random.py")


def run(args: argparse.Namespace) -> None:
    raise NotImplementedError("handler_X is a stub — see handler_random.py")
