"""Random CVRP instance generation."""
import torch
from tensordict import TensorDict


def generate_random_instances(
    n_problems: int,
    dim: int,
    seed: int,
    device: torch.device,
    max_load: int = 30,
) -> TensorDict:
    """
    Generate a batch of random CVRP instances.

    Args:
        n_problems: Number of parallel instances (B).
        dim: Number of customers (N). Depot is added at index 0.
        seed: RNG seed for reproducibility.
        device: Target device.
        max_load: Vehicle capacity (Q).

    Returns:
        TensorDict with batch_size=[n_problems], keys:
          coords:   (B, N+1, 2)  float, uniform in [0, 1]^2
          demands:  (B, N+1)     int64, in {1, ..., 9}; depot has demand 0
          capacity: (B, 1)       int64, all max_load
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    coords = torch.rand(n_problems, dim + 1, 2, generator=gen, device=device)
    demands = torch.randint(
        1, 10, (n_problems, dim + 1), generator=gen, device=device
    )
    demands[:, 0] = 0
    capacity = torch.full((n_problems, 1), max_load, dtype=torch.int64, device=device)
    return TensorDict(
        {"coords": coords, "demands": demands, "capacity": capacity},
        batch_size=[n_problems],
        device=device,
    )
