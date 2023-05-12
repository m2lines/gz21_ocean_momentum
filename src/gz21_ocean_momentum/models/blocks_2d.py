"""Simple two-dimensional blocks."""
from typing import List

from torch.nn import Sequential, Module, Conv2d, ReLU, BatchNorm2d


class ConvBlock(Sequential):
    """Simple two-dimensional convolutional block.

    Conv2d -> ReLU -> BatchNorm2d (optional).

    Parameters
    ----------
    in_chans : int
        The number of input channels the block should take.
    out_chans : int
        The number of input channels the block should produce.
    kernal_size : int
        The size (in pixels) of the convolutional kernel.
    padding : int
        The size of the padding to apply to the output.
    bnorm : bool
        Bool determining whether batch normalisation is included or not.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        pad: int,
        bnorm: bool,
    ):
        """Build ``ConvBlock``."""
        super().__init__(
            *self._layers(
                _process_positive_int(in_chans),
                _process_positive_int(out_chans),
                _process_positive_int(kernel_size),
                _process_positive_int(pad),
                _process_bool_arg(bnorm),
            )
        )

    @staticmethod
    def _layers(
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        pad: int,
        bnorm: bool,
    ) -> List[Module]:
        """Return a list of the block's layers.

        Parameters
        ----------
        in_chans : int
            Number of input channels the block should expect.
        out_chans : int
            The number of output channels the block should produce.
        kernel_size : intm
            Size of the convolutional kernel.
        pad : int
            The amount of padding to apply to the output.
        bnorm : int
            Whether or not a ``BatchNorm2d`` is included.

        Returns
        -------
        layer_list : List[Module]
            A list of the layers in the block.

        """
        layer_list = [
            Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=pad),
            ReLU(),
        ]
        if bnorm is True:
            layer_list.append(BatchNorm2d(out_chans))
        return layer_list


def _process_positive_int(positive_int) -> int:
    """Process ``positive_int`` argument.

    Parameters
    ----------
    positive_int : int
        Argument to process.

    Returns
    -------
    positive_int : int
        See Parameters.

    Raises
    ------
    TypeError
        If ``positive_int`` is not an integer.
    ValueError
        If ``positive_int`` is not positive.

    """
    if not isinstance(positive_int, int):
        msg = f"Expected '{positive_int}' to be int. Got "
        msg += f"'{type(positive_int)}'."
        raise TypeError(msg)
    if positive_int < 0:
        msg = f"Expected positive int. Got '{positive_int}'."
        raise ValueError(positive_int)

    return positive_int


def _process_bool_arg(bool_arg: True) -> bool:
    """Process ``bool_arg``.

    Parameters
    ----------
    bool_arg : bool
        Boolean argument to process.

    Returns
    -------
    bool_arg : bool
        See Parameters.

    Raises
    ------
    TypeError
        If ``bool_arg`` is not a bool

    """
    if not isinstance(bool_arg, bool):
        raise TypeError(f"Expected boolean arg. Got '{type(bool_arg)}'.")

    return bool_arg
