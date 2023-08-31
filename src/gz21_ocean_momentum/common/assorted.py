def list_is_strictly_increasing(xs: list[float]) -> bool:
    """
    Is this list monotonically increasing? Does not permit repeated elements.

    Asserts that a list is in the correct format to be consumed by the
    `milestones` parameter in `torch.optim.MultiStepLR(optimizer: list, ...)`.
    """
    return all(xl<xr for xl, xr in zip(xs, xs[1:]))
