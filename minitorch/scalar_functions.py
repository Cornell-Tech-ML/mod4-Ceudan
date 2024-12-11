from __future__ import annotations

# from re import T
from typing import TYPE_CHECKING

# from networkx import sigma

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Applies backward pass on specific child class function."""
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Applies forward pass on specific child class function."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the specific child class function to the given variables."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        c = float(c)  # I HAD TO ADD THIS LINE TO PASS THE BOOL TESTS
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function f(x, y) = x + y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return the sum of a and b"""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivative of addition is 1"""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function f(x) = log(x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the log of a"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of the log function is 1/x"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.
# TODO: Implement for Task 1.2.


class Mul(ScalarFunction):
    """Multiply Function f(x,y) = x*y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return the product of a and b"""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivative of the multiplication function is the other variable"""
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Performs the Inverse Operation"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the inverse of a"""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of the inverse function is -1/x^2"""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negate Function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the negation of a"""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of the negation function is -1"""
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid Function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the sigmoid of a"""
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of the sigmoid function"""
        s: float = ctx.saved_values[0]
        return s * (1.0 - s) * d_output


class ReLU(ScalarFunction):
    """ReLU Function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the ReLU of a"""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of the ReLU function is 1 if x >= 0 else 0"""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential Function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return the exponential of a"""
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of the exponential function is itself"""
        out: float = ctx.saved_values[0]
        return out * d_output


class EQ(ScalarFunction):
    """Equal Function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return True if a is equal to b"""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivative of the equal function is 0"""
        return 0.0, 0.0


class LT(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return True if a is less than b"""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivative of the less than function is 0"""
        return 0.0, 0.0
