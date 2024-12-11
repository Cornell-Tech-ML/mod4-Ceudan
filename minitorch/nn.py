from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    # # renamed the variables for interpretability
    batch_size, nchals, nrows, ncols = input.shape
    nkrows, nkcols = kernel
    assert nrows % nkrows == 0
    assert ncols % nkcols == 0
    # TODO: Implement for Task 4.3.
    # each kernel operation produces a single output
    new_nrows = nrows // nkrows
    new_ncols = ncols // nkcols
    # view does most of our work for us
    out = (
        input.contiguous()
        .view(batch_size, nchals, nrows, new_ncols, nkcols)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch_size, nchals, new_nrows, new_ncols, nkrows * nkcols)
    )
    return out, new_nrows, new_ncols


# TODO: Implement for Task 4.3.
custom_max = FastOps.reduce(operators.max, float("-inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Return a 1-hot tensor with the max index set to 1.

    Args:
    ----
        input: Tensor in question
        dim: Dimension to compute argmax over

    Returns:
    -------
        A 1-hot tensor with the same shape as `input`. All values are 0 or 1. A 1 signifies it was the max value along the specified dimension.

    """
    # get the max vals
    max_vals = custom_max(input, dim)
    # create the binary mask
    out = max_vals == input
    return out


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies average pooling with 2d kernel.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    # reshape it
    input_mod, new_nrow, new_ncols = tile(input, kernel)
    # mean over the temporary dimension
    out = input_mod.mean(dim=4)
    # remove the unneeded dimension
    out = out.view(input.shape[0], input.shape[1], new_nrow, new_ncols)
    return out


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Get the max value along a dimension specified dimension."""
        # use prexisting max function
        max_vals = custom_max(input, int(dim.item()))
        ctx.save_for_backward(input, dim)
        return max_vals

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient"""
        input, dim = ctx.saved_values
        # argmax give us a nice mask
        grad = grad_output * argmax(input, int(dim.item()))
        return grad, 0.0


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along a specific dimension."""
    # we subtract the max value to avoid numerical instability
    stabilizer_vals = max(input, dim)
    ins_exp = (input - stabilizer_vals).exp()
    den = ins_exp.sum(dim)
    out = ins_exp / den
    return out


def _max(input: Tensor, dim: int) -> Tensor:
    """Apply the max reduction operation."""
    out = Max.apply(input, input._ensure_tensor(dim))
    return out


max = _max


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along a specific dimension."""
    stabilizer_vals = max(input, dim)
    ins_exp = (input - stabilizer_vals).exp()
    den = ins_exp.sum(dim)
    den_log = den.log()

    return input - den_log - stabilizer_vals


def dropout(input: Tensor, p: float = 0.5, ignore: bool = False) -> Tensor:
    """Perform the dropout operations which zeros out random elements of the input tensor."""
    if ignore:
        # no dropoout
        out = input
    else:
        # rand generates random values uniform in range [0, 1), hence we can threshold on that
        filt = rand(input.shape) > p
        out = input * filt
    return out


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform max pooling with a 2d kernel."""
    # reshape the input
    input_mod, new_nr, new_nc = tile(input, kernel)
    # max over the temporary dimension
    out = max(input_mod, dim=4)
    # remove the unneeded dimension
    out = out.view(input.shape[0], input.shape[1], new_nr, new_nc)
    return out
