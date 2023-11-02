# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:40:09 2020

@author: Arthur
"""
from enum import Enum


class DEVICE_TYPE(Enum):
    GPU = "GPU"
    CPU = "CPU"


def print_every(to_print: str, every: int, n_iter: int) -> bool:
    """Prints every given number of iterations.

    Parameters
    ----------

    :to_print: str,
        The string to print

    :every: int,
        The string passed to the function is only printed every 'every' call.

    :n_iter: int,
        The index of the calls, which is to be handled by the user.

    Returns
    ----------
    Bool
        True if printed, False if not
    """
    if n_iter % every == every - 1:
        print(to_print)
        return True
    return False


class RunningAverage:
    """Class for online computing of a running average"""

    def __init__(self) -> None:
        self.n_items = 0
        self.average = 0.0

    @property
    def value(self) -> float:
        return self.average

    def update(self, value: float, weight: int = 1) -> float:
        """Adds some value to be used in the running average.

        Parameters
        ----------

        :value: float,
            Value to be added in the computation of the running
            average.

        :weight: int,
            Weight to be given to the passed value.
            Can be useful if the function
            update is called with values that already are averages over some
            given number of elements.

        Returns
        -------
        The updated value of the average

        Examples
        --------
        blablabla
        """
        temp = self.average * self.n_items + value * weight
        self.n_items = self.n_items + weight
        self.average = temp / self.n_items
        return self.average

    def reset(self) -> None:
        """Resets the running average to zero as well as its number of items"""
        self.n_items = 0
        self.average = 0.0

    def __str__(self) -> str:
        return str(self.average)


def learning_rates_from_string(rates_string: str) -> dict[int, float]:
    temp = rates_string.split("/")
    if len(temp) == 1:
        return {0: float(rates_string)}
    if len(temp) % 2 != 0:
        raise Exception("The learning rates should be provided in pairs.")
    rates = {}
    for i in range(int(len(temp) / 2)):
        rates[int(temp[2 * i])] = float(temp[2 * i + 1])
    return rates


def run_ids_from_string(run_ids_str: str) -> list[str]:
    return run_ids_str.split("/")


def list_from_string(string: str) -> list[str]:
    return string.split("/")
