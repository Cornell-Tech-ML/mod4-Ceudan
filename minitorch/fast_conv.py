from typing import Tuple, TypeVar, Any

from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # TODO: Implement for Task 4.1.
    for bind in prange(batch_):
        # save to local thread
        s1 = input_strides
        s2 = weight_strides
        # iterate over each output
        for oc_ind in range(out_channels):
            for ow_ind in range(out_width):
                # now we are calculating the output value of a particular output
                oval = 0.0  # accucmulated value
                for ic_ind in range(in_channels):
                    for kw_ind in range(kw):
                        if reverse:
                            in_pos = ow_ind - kw + 1 + kw_ind
                        else:
                            in_pos = ow_ind + kw_ind
                        if 0 <= in_pos < width:
                            # in pos is jumpy by batch number, jump by in_channel number, jump by position in kernel
                            input_index = (
                                (bind * s1[0]) + (ic_ind * s1[1]) + (in_pos * s1[2])
                            )
                            # weight index is jumpy by out_channel number, jump by in_channel number, jump by position in kernel
                            weight_index = (
                                (oc_ind * s2[0]) + (ic_ind * s2[1]) + (kw_ind * s2[2])
                            )
                            oval += input[input_index] * weight[weight_index]

                out_index = (
                    (bind * out_strides[0])
                    + (oc_ind * out_strides[1])
                    + (ow_ind * out_strides[2])
                )
                out[out_index] = oval


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    # batch_, out_channels, _, _ = out_shape
    # batch, in_channels, height, width = input_shape
    # out_channels_, in_channels_, kh, kw = weight_shape

    # I took more info from the outshape.
    batch_, out_channels, out_nrows, out_ncols = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # I renamed the varibales for interpretability
    # input_strides_batch, input_strides_channel, input_strides_row, input_strides_column
    isb, isc, isr, isc = s1[0], s1[1], s1[2], s1[3]
    wsoc, wsic, wsr, wsc = s2[0], s2[1], s2[2], s2[3]
    osb, osc, osr, osc = out_strides

    # TODO: Implement for Task 4.2.
    # we manually iterate over the output indices in the output tensor
    # and then go over the input tensor and add them
    for bi in prange(batch):
        for oci in range(out_channels):
            for ohi in range(out_nrows):
                for owi in range(out_ncols):
                    out_val = 0.0  # output value at this particular index
                    # now we iterate over positions in the input matrix in the receptive field of this output position
                    for ici in range(in_channels):
                        for kri in range(kh):
                            for kci in range(kw):
                                if reverse:
                                    iri = ohi - kh + 1 + kri
                                    icoi = owi - kw + 1 + kci
                                else:
                                    iri = ohi + kri
                                    icoi = owi + kci

                                # actually we have no padding so just ignore out of bounds computations
                                if 0 <= iri < height and 0 <= icoi < width:
                                    # get the "real" position in the tensors
                                    input_idx = (
                                        (bi * isb)
                                        + (ici * isc)
                                        + (iri * isr)
                                        + (icoi * isc)
                                    )
                                    weight_idx = (
                                        (oci * wsoc)
                                        + (ici * wsic)
                                        + (kri * wsr)
                                        + (kci * wsc)
                                    )
                                    # dot product
                                    out_val += input[input_idx] * weight[weight_idx]

                    # Compute flat index for the output tensor and store result
                    out_idx = bi * osb + oci * osc + ohi * osr + owi * osc
                    out[out_idx] = out_val


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
