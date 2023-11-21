def list_is_strictly_increasing(xs: list) -> bool:
    """
    Is this list monotonically increasing? Does not permit repeated elements.
    List elements must be orderable.

    Asserts that a list is in the correct format to be consumed by the
    `milestones` parameter in `torch.optim.MultiStepLR(optimizer: list, ...)`.
    """
    return all(xl<xr for xl, xr in zip(xs, xs[1:]))

def at_idx_pct(pct: float, a) -> int:
    """
    Obtain the index into the given list-like to the given percent.
    No interpolation is performed: we choose the leftmost closest index i.e. the
    result is floored.

    e.g. `at_idx_pct(0.5, [0,1,2]) == 1`

    Must be able to `len(a)`.

    Invariant: `0<=pct<=1`.

    Returns a valid index into `a`.
    """
    return int(pct * len(a))
