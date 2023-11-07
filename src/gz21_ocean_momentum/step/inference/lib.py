def cm2_6_prep_pytorch(_: xr.Dataset, idx_start: float, idx_end) -> _:
    """
    Various transformations, subsetting of a CM2.6 dataset.

    Retrieve using data step lib, slice & restrict spatial domain.

    idx_start: 0->1 subset of dataset to use, start
    idx_end:   0->1 subset of dataset to use, end (must be > idx_start)
    """
