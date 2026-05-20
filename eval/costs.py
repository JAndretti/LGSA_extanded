"""Cost-function stubs for future benchmark conventions.

The `random` handler doesn't use this file — it reads costs via
`src.cvrp.state.cost`, which is exact Euclidean over normalized coords.

To be implemented when handler_X is added:
  - cvrplib_rounded_cost(solution, raw_coords) -> int   # CVRPLib X/XL convention
  - exact_euclidean_cost(solution, raw_coords) -> float # Queiroga XML convention
  - extract_and_cost(solution, actual_N, raw_coords, rounded) -> float
"""
