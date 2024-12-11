# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        if (
            i < out_size
        ):  # each i represents position in out memory, if out of bounds do nothing
            # we convert the position in out memory to an index in input tensor
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(
                in_index, in_strides
            )  # calculate ordinal position for in
            out_pos = index_to_position(out_index, out_strides)
            # print(i,out_position)

            # you have out and in locations in memory, start mapping
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if (
            i < out_size
        ):  # each i represents position in out memory, if out of bounds do nothing
            # we get the positions in a and b memory that map to out
            to_index(i, out_shape, out_index)
            # get a and b indexes
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # get a and b memory positions
            pos_a = index_to_position(a_index, a_strides)
            pos_b = index_to_position(b_index, b_strides)
            pos_o = index_to_position(out_index, out_strides)

            # 7 reads (not counting reads inside above functions) 1 write to global memory
            out[pos_o] = fn(a_storage[pos_a], b_storage[pos_b])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")

    # copy over our block's assigned section of orignal array to our block's shared array
    if i < size:
        cache[pos] = a[i]
    else:
        # this block has spilled over out of bounds of the original array
        cache[pos] = 0
    cuda.syncthreads()

    # sum all elements
    if pos % 2 == 0:
        cache[pos] += cache[pos + 1]
    cuda.syncthreads()
    if pos % 4 == 0:
        cache[pos] += cache[pos + 2]
    cuda.syncthreads()
    if pos % 8 == 0:
        cache[pos] += cache[pos + 4]
    cuda.syncthreads()
    if pos % 16 == 0:
        cache[pos] += cache[pos + 8]
    cuda.syncthreads()
    if pos % 32 == 0:
        cache[pos] += cache[pos + 16]
    cuda.syncthreads()
    # We stop here since the next index of 32 would spill over

    # return answer for each block
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]  # store result in out


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        # copy out key data to local memory
        red_size = a_shape[reduce_dim]
        red_stride = a_strides[reduce_dim]

        if (
            out_pos < out_size
        ):  # out pos is a position in the output memory, that is the result of reducing out the reduce dimension
            #  base_a_idx is the 0 along the reduce dim in the input a array
            to_index(out_pos, out_shape, out_index)
            base_a_idx = index_to_position(out_index, a_strides)

            # we load the entire length along the reduce dim into our shared memoy
            if pos < red_size:  # pos might overflow reduce dim size
                cache[pos] = fn(
                    reduce_value, a_storage[base_a_idx + pos * red_stride]
                )  # we move along reduce dimension by jumping by its stride
            else:
                cache[pos] = (
                    reduce_value  # fill valus that should not exist with values that don't interfere
                )
            cuda.syncthreads()  # synchronize threads

            stage = 1  # This is how high we are in a hypothetical binary sum tree
            while (
                stage < BLOCK_DIM
            ):  # just sum over entire block, don't think about where the actual and artificial real values lie
                if pos % (2 * stage) == 0:
                    cache[pos] = fn(cache[pos], cache[pos + stage])  # sum
                stage = stage * 2
                cuda.syncthreads()  # synchronize threads

            if pos == 0:
                out[out_pos] = cache[0]  # store result in out

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
    # I assume that the size of our block is greater or equal to the size of all 3 square matrices
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    if local_i < size and local_j < size:  # this has to change for the full matrix mult
        # transfer information into shared memory
        shared_a[local_i, local_j] = a[local_i * size + local_j]
        shared_b[local_i, local_j] = b[local_i * size + local_j]

        sum = (
            0.0  # accumulate sum inside shared memory to avoid excessive global writes
        )
        cuda.syncthreads()  # always sync before you access shared memory
        for n in range(
            size
        ):  # we are iterating over the dimension that will dissappear
            sum += shared_a[local_i, n] * shared_b[n, local_j]
        out[local_i * size + local_j] = (
            sum  # just write to out once, not inside the loop
        )


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    r = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    c = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    lr = cuda.threadIdx.x
    lc = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # raise NotImplementedError("Need to implement for Task 3.4")
    # shape (B, I, K), (B, K, J)
    # Number of tiles along the K dimension
    ANCBNR = a_shape[-1]  # number if cols in A which is number of rows in B
    BNC = b_shape[-1]  # number of cols in B
    ANR = a_shape[-2]  # number of rows in A

    # we only access the original a and b at the given number
    a_batch_offset = batch * a_batch_stride
    b_batch_offset = batch * b_batch_stride

    # we need these values to jump along the axis
    AR_strides = a_strides[-2]
    AC_strides = a_strides[-1]
    BR_strides = b_strides[-2]
    BC_strides = b_strides[-1]

    sum = 0.0  # accumulate the dot product sum outside of the global variable
    for tile_f_pos in range(
        0, ANCBNR, BLOCK_DIM
    ):  # this is the offset provided by which tile_number we are on
        # the positions we are in the a and b
        ac = tile_f_pos + lc
        br = tile_f_pos + lr

        # load into shared memory if you have a valid source position
        if r < ANR and ac < ANCBNR:
            a_pos = a_batch_offset + r * AR_strides + ac * AC_strides
            a_shared[lr, lc] = a_storage[a_pos]
        else:
            a_shared[lr, lc] = (
                0.0  # this value should not interfere in later dot product
            )
        if br < ANCBNR and c < BNC:
            b_pos = b_batch_offset + br * BR_strides + c * BC_strides
            b_shared[lr, lc] = b_storage[b_pos]
        else:
            b_shared[lr, lc] = 0.0
        cuda.syncthreads()  # don't access till everythread is done loading into shared memory

        # we perform the dot product by moving across the dimension ANCBNR
        for k in range(
            min(BLOCK_DIM, ANCBNR - tile_f_pos)
        ):  # we could just stop at block dim, since we would add 0*0 in the invalid shared positions, but remember speed
            sum += a_shared[lr, k] * b_shared[k, lc]
        cuda.syncthreads()

    # we add solution to the rightful position in out matrix
    if r < ANR and c < BNC:
        # Calculate the memory address for the output
        out_batch_offset = batch * out_strides[0]
        out_pos = out_batch_offset + r * out_strides[-2] + c * out_strides[-1]
        out[out_pos] = sum


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
