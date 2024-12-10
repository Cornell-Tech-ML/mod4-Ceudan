# from minitorch.scalar import Scalar # commented out to avoid circular import
# ## Task 1.1
# Central Difference calculation

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol

def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    vals1 = [v for v in vals]

    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    history: Any

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates sets the derivative for the leaf node in computation graph."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant (no history)"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of this variable in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the derivate of this variable with respect to its inputs in its last function call, multiplied by `d_output`."""
        ...


# def topological_recursion(
#     variable: Variable, visited: set[int], order: List[Variable]
# ) -> None:
#     """Recursively visits all the children of the variable in a topological order.

#     Arguments:
#     ---------
#         variable (Variable): The current variable
#         visited (set[int]): A set of visited variables
#         order (List[Variable]): The order of the variables to calculate in topological order

#     Returns:
#     -------
#         None

#     """
#     if variable.unique_id in visited:  # already visited
#         return
#     visited.add(variable.unique_id)  # mark as visited
#     # print("VARIABLE", variable)
#     # print("VARIABLE TYPE", type(variable))
#     # print("VARIABLE.HISTORY", variable.history)
#     if variable.history is not None:
#         children = variable.history.inputs
#         for child in children:
#             topological_recursion(child, visited, order)
#         order.append(variable)
"""ABOVE WAS MY PERSONAL IMPLEMENTATION"""


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # # TODO: Implement for Task 1.4.
    # visited = set()
    # order = []
    # topological_recursion(variable, visited, order)
    """ABOVE WAS MY PERSONAL IMPLEMENTATION"""
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
         None: No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    # # print("Starting backpropop from VARIABLE:", variable)
    # order = topological_sort(variable)
    # derivatives = {}  # store the derivative of loss to each variable
    # derivatives[variable.unique_id] = (
    #     deriv  # the derivative of loss to the rightmost parent variable
    # )
    # for var in order:
    #     if var.is_leaf():
    #         # should be called only after all parents have been visited
    #         var.accumulate_derivative(derivatives[var.unique_id])
    #     else:
    #         # compute the derivative of loss to each child of current variable
    #         d_output = derivatives[
    #             var.unique_id
    #         ]  # the derivative of loss to current variable
    #         child_ders = var.chain_rule(d_output)  # list of tuples of (child, derivate)
    #         for child, der in child_ders:
    #             if child.unique_id in derivatives:
    #                 derivatives[child.unique_id] += der
    #             else:
    #                 derivatives[child.unique_id] = der
    """ABOVE WAS MY PERSONAL IMPLEMENTATION"""
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values of the context."""
        return self.saved_values
