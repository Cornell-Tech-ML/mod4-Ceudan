"""Collection of the core mathematical operators used throughout the code base."""

import math


# ## Task 0.1
from typing import Callable, Iterable, TypeVar

# Implementation of a prelude of elementary functions.
# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The product of `x` and `y`.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The input number `x` unchanged.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The sum of `x` and `y`.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The negated value of `x`.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: `True` if `x` is less than `y`, otherwise `False`.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: `True` if `x` is equal to `y`, otherwise `False`.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The larger of `x` and `y`.

    """
    return x if x > y else y


def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Checks if two numbers are close in value within a tolerance.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.
        tol (float, optional): The tolerance level. Defaults to 1e-9.

    Returns:
    -------
        bool: `True` if `x` and `y` are close within the tolerance, otherwise `False`.

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The sigmoid of `x`, defined as 1 / (1 + exp(-x)).

    """
    if x >= 0:
        res = 1.0 / (1.0 + math.exp(-x))
    else:
        res = math.exp(x) / (1.0 + math.exp(x))
    return float(res)


def relu(x: float) -> float:
    """Applies the ReLU (Rectified Linear Unit) activation function.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The ReLU of `x`, which is `x` if `x > 0`, otherwise `0`.

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm of a number.

    Args:
    ----
        x (float): The input number. Must be positive.

    Returns:
    -------
        float: The natural logarithm of `x`.

    Raises:
    ------
        ValueError: If `x` is less than or equal to 0.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The exponential of `x`, defined as e^x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal of a number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The reciprocal of `x`, defined as 1 / x.

    Raises:
    ------
        ZeroDivisionError: If `x` is zero.

    """
    return 1.0 / x


def log_back(x: float, a: float) -> float:
    """Computes the derivative of the logarithm function times a second argument.

    Args:
    ----
        x (float): The input to the logarithm function.
        a (float): A constant to multiply to the derivate of the log function

    Returns:
    -------
        float: The derivative of the logarithm function times `a`.

    Raises:
    ------
        ValueError: If `x` is less than or equal to 0.

    """
    return a / (x + EPS)


def inv_back(x: float, a: float) -> float:
    """Computes the derivative of the reciprocal function times a second argument.

    Args:
    ----
        x (float): The input to the reciprocal function.
        a (float): The derivative of the output with respect to the reciprocal's output.

    Returns:
    -------
        float: The derivative of the reciprocal function times `a`.

    """
    return -(1.0 / x**2) * a


def relu_back(x: float, a: float) -> float:
    """Computes the derivative of the ReLU function times a second argument.

    Args:
    ----
        x (float): The input to the ReLU function.
        a (float): The derivative of the output with respect to ReLU's output.

    Returns:
    -------
        float: The derivative of ReLU function times `a`.

    """
    return a if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
X = TypeVar("X")
N = TypeVar("N", float, int)


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
    fn: Function from one value to one value.

    Returns:
    -------
    A function that takes a list, applies “fn* to each element, and returns a
    new list

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
    fn: combine two values

    Returns:
    -------
    Function that takes two equally sized lists “lsl* and “1ls2°, produce a new list
    by applying fn(x, y) on each pair of elements.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce.

    Args:
    ----
    fn: combine two values
    start: start value $x_0$

    Returns:
    -------
    Function that takes a list “ls* of elements
    $x_1 \ldots x_n$ and computes the reduction :math:*fn(x_3, fn(x_2,
    fn(x_1, x_0)))°

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negates all elements in a list using the `map` function.

    Args:
    ----
        lst (List[float]): A list of floats.

    Returns:
    -------
        List[float]: A list with all elements negated.

    """
    return map(neg)(lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements from two lists using the `zipWith` function.

    Args:
    ----
        lst1 (List[float]): The first list of floats.
        lst2 (List[float]): The second list of floats.

    Returns:
    -------
        List[float]: A list of floats where each element is the sum of the corresponding elements in `lst1` and `lst2`.

    """
    return zipWith(add)(lst1, lst2)


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using ‘reduce’ and ‘add*."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Broduct of a list using “reduce’ and ‘mul*."""
    return reduce(mul, 1.0)(ls)
