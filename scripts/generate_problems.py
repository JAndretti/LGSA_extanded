"""Pre-generate CVRP problem batches and save them as .pt files.

One file per (dim, capacity) pair. Capacity follows the Nazari convention by
default but can be overridden by editing _CAPACITY_BY_DIM.

Files are saved to OUTPUT_DIR (default: generated_problems/) at the project
root, with names like `gen_dim{dim}_load{capacity}.pt`. The loader at
`src.data.loader.load_eval_instances` reads them back keyed by (dim, max_load).
"""
from pathlib import Path

import torch
from loguru import logger


N_PROBLEMS = 10_000
DIMS = [10, 20, 50, 100, 500, 1000]
DEVICE = "cpu"
SEED = 1234

# Standard Nazari capacities per problem size.
_CAPACITY_BY_DIM = {10: 20, 20: 30, 50: 40, 100: 50, 500: 50, 1000: 50}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "generated_problems"


def generate_and_save(
    n_problems: int,
    dim: int,
    capacity: int,
    out_dir: Path,
    seed: int,
    device: str = "cpu",
) -> Path:
    gen = torch.Generator(device=device).manual_seed(seed)
    coordinates = torch.rand(n_problems, dim + 1, 2, generator=gen, device=device)
    demands = torch.randint(
        1, 10, (n_problems, dim + 1), generator=gen, device=device
    )
    demands[:, 0] = 0
    capacities = torch.full(
        (n_problems, 1), capacity, dtype=torch.int64, device=device
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gen_dim{dim}_load{capacity}.pt"
    torch.save(
        {
            "node_coords": coordinates,
            "demands": demands,
            "capacity": capacities,
        },
        path,
    )
    return path


def main():
    for dim in DIMS:
        capacity = _CAPACITY_BY_DIM.get(dim, 50)
        path = generate_and_save(
            n_problems=N_PROBLEMS,
            dim=dim,
            capacity=capacity,
            out_dir=OUTPUT_DIR,
            seed=SEED + dim,    # disjoint seed per dim
            device=DEVICE,
        )
        logger.info(
            f"Saved {N_PROBLEMS} problems | dim={dim} | capacity={capacity} -> {path}"
        )


if __name__ == "__main__":
    main()
