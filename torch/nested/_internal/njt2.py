import torch
import itertools
from torch import Tensor
from typing import List, Optional, Tuple

NJT_OPS = {}


nid = itertools.count()

class NestedInt:
    def __init__(self, offsets):
        self._id = next(nid)
        self._offsets = offsets

    def __repr__(self):
        return f"j{self._id}"

# TODO(rzou: weak key dictionary
NESTED_INT_CACHE = {}

class NJT2:
    def __init__(self, values, offsets, _ragged_idx=1, _metadata_cache=None):
        self._values = values
        self._offsets = offsets
        assert self._values.device == self._offsets.device
        if self._offsets not in NESTED_INT_CACHE:
            NESTED_INT_CACHE[self._offsets] = NestedInt(self._offsets)
        self._nested_int = NESTED_INT_CACHE[self._offsets]
        self._ragged_idx = _ragged_idx
        self._metadata_cache = None  # for compatibility

    @property
    def __class__(self):
        # TODO(rzou): is this bad?
        return torch.Tensor

    def values(self):
        return self._values

    def offsets(self):
        return self._offsets

    def lengths(self):
        return self._lengths
    
    @property
    def _lengths(self):
        return None

    def requires_grad_(self, requires_grad):
        self._values.requires_grad_(requires_grad)

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def device(self):
        return self._values.device

    @property
    def shape(self):
        return (self._offsets.shape[0] - 1, self._nested_int, self._values.shape[1:])

    @property
    def _size(self):
        return self.shape

    @property
    def requires_grad(self):
        return self._values.requires_grad

    @property
    def layout(self):
        return torch.jagged

    def size(self):
        return self.shape

    def dim(self):
        return self._values.dim() + 1

    def is_contiguous(self):
        return self._values.is_contiguous()

    @property
    def is_nested(self):
        return True

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in NJT_OPS:
            return NJT_OPS[func](func, *args, **kwargs)
        raise RuntimeError(f"NYI: {func}")

    # Method-only are annoying
    def to(self, *args, **kwargs):
        if 'device' in kwargs:
            return njt_like(self, self._values.to(*args, **kwargs), self._offsets.to(kwargs["device"]))
        else:
            return njt_like(self, self._values.to(*args, **kwargs))

    def __mul__(self, other):
        return torch.mul(self, other)

    def __add__(self, other):
        return torch.add(self, other)

    def __sub__(self, other):
        return torch.sub(self, other)

    @property
    def grad(self):
        return njt_like(self, self._values.grad)

    def backward(self, grad_output=None):
        if grad_output is not None and isinstance(grad_output, torch.Tensor):
            assert same_raggedness(grad_output, self)
            self._values.backward(grad_output._values)
        else:
            self._values.backward(grad_output)


def same_raggedness(a, b):
    return a._offsets is b._offsets and a._ragged_idx == b._ragged_idx


# Not going to work for in-place
def bind_method(api):
    if isinstance(api, str):
        func = getattr(torch, api)
        def method(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        setattr(NJT2, api, method)
    else:
        for a in api:
            bind_method(a)

bind_method([
    "unbind",
    "sin",
    "cos",
    "neg",
    "abs",
    "transpose",
    "chunk",
])

def register_njt2(func):
    if callable(func):
        def inner(impl):
            NJT_OPS[func] = impl
            return impl
        return inner
    assert isinstance(func, list)
    def inner(impl):
        for f in func:
            NJT_OPS[f] = impl
        return impl
    return inner

# returns True if the raggedness-relevant portions of the NT shape
# match those of the specified size
def raggedness_matches(nt, size):
    end = nt._ragged_idx + 1
    nt_ragged = nt._size[:end]
    size_ragged = size[:end]
    return len(nt_ragged) == len(size_ragged) and (
        all(ns == s or s == -1 for ns, s in zip(nt_ragged, size_ragged))
    )

def njt_like(a, values=None, offsets=None, ragged_idx=None):
    if values is None:
        values = a._values
    if offsets is None:
        offsets = a._offsets
    if ragged_idx is None:
        ragged_idx = a._ragged_idx
    return NJT2(values, offsets, ragged_idx)

def extract_kwargs(arg):
    kwargs = {
        "offsets": arg.offsets(),
        "_ragged_idx": arg._ragged_idx,
    }
    return kwargs

# binary pointwise
@register_njt2([
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
])
def jagged_binary_pointwise(func, input, other, *, out=None, **kwargs):
    a = input
    b = other
    assert isinstance(a, NJT2) or isinstance(b, NJT2)
    assert out is None

    if not isinstance(other, torch.Tensor):
        # unary case
        return njt_like(input, func(input._values, other, **kwargs))

    mismatch_error_msg = (
        "cannot call binary pointwise function {} with inputs of shapes {} and {}"
    )
    # a is NT, b is NT
    if isinstance(a, NJT2) and isinstance(b, NJT2):
        # ex: (B, j0, D) + (B, j0, D)
        # ex: (B, j0, D) + (B, j0, 1)
        if raggedness_matches(a, b._size):
            return njt_like(
                a,
                func(a._values, b._values, **kwargs)
            )
        raise RuntimeError(mismatch_error_msg.format(func.__name__, a._size, b._size))
    # either a is NT or b is NT at this point
    a_is_nt = isinstance(a, NJT2)
    extracted_kwargs = extract_kwargs(a) if a_is_nt else extract_kwargs(b)

    # === Handle broadcasting across the batch / ragged dims ===

    # Easy case: take advantage of pre-existing broadcasting logic
    # ex: (B, j0, ?, ?) + (?) -> (B, j0, ?, ?)
    # ex: (B, j0, ?, ?) + (?, ?) -> (B, j0, ?, ?)
    # ex: (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    nt, t = (a, b) if a_is_nt else (b, a)
    # See Note: [ Squeezing leading ones ]
    if t.dim() > nt.dim():
        raise NotImplementedError("NYI: broadcasting NT with T with larger dim")
    t_squeezed = squeeze_leading_ones(t)
    if nt.dim() >= t_squeezed.dim() + 2:
        lhs, rhs = (nt._values, t_squeezed) if a_is_nt else (t_squeezed, nt._values)
        return NJT2(func(lhs, rhs, **kwargs), **extracted_kwargs)

    # Harder case: do manual broadcasting over unbound components
    # when NT dim == non-NT dim
    # ex: (B, j0, D_0, D_1) + (B, 1, D_0, D_1) -> (B, j0, D_0, D_1)
    if a.dim() == b.dim():
        # ex: (B, j0, D_0, D_1) + (1, 1, D_0, D_1) -> should
        # be (B, j0, D_0, D_1) but not yet supported
        if a.shape[0] != b.shape[0]:
            raise RuntimeError(
                mismatch_error_msg.format(func.__name__, a.shape, b.shape)
            )

        # need to use offsets to broadcast across ragged dim properly
        # NB: inefficient fallback here; Triton codegen can help this
        # TODO: Make this work with autograd
        outputs = []
        for a_comp, b_comp in zip(a.unbind(), b.unbind()):
            outputs.append(func(a_comp, b_comp, **kwargs))
        new_values = torch.cat(outputs, dim=0)
        return NJT2(new_values, **extracted_kwargs)

import math
import torch.nn.functional as F

@register_njt2(torch.chunk)
def chunk_default(func, inputs, chunks, dim=0):
    new_kwargs = {
        "input": inputs,
        "chunks": chunks,
        "dim": dim,
    }

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], "chunk", allow_batch_dim=True
    )

    if new_kwargs["dim"] == 0:
        chunks = new_kwargs["chunks"]
        dim0_size = inp._size[0]
        chunk_size = math.ceil(dim0_size / chunks)

        # get _offsets of the chunks
        lengths = inp._offsets.diff()
        chunked_lengths = lengths.chunk(chunks)
        chunked_offsets = [torch.cumsum(x, dim=0) for x in chunked_lengths]
        chunked_offsets = [F.pad(x, (1, 0), value=0) for x in chunked_offsets]
        nested_kwargs = [
            {"offsets": per_offsets, "ragged_idx": inp._ragged_idx}
            for per_offsets in chunked_offsets
        ]

        # get _values of the chunks
        split_sizes = [x.sum().item() for x in chunked_lengths]
        chunk_values = inp._values.split(split_sizes)

        return [
            NJT2(values=chunk_values[i], **(nested_kwargs[i]))
            for i in range(0, chunk_size)
        ]
    else:
        return [
            NJT2(values=x, **extract_kwargs(inp))
            for x in func(inp._values, **new_kwargs)
        ]


def squeeze_leading_ones(t):
    # Note: [ Squeezing leading ones ]
    #
    # Squeeze leading ones from t.
    #
    # We want:
    #   (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    #   (B, j0, ?, ?) + (1, 1, 1, ?, ?) -> (1, B, j0, ?, ?)  (not yet supported)
    #
    # 1) Squeeze extra ones and grab values from NT
    #   (1, 1, ?, ?) -> (?, ?)   and   (sum(*), ?, ?) -> (B, j0, ?, ?)
    # 2) Do dense broadcasting:
    #   (sum(*), ?, ?) + (?, ?) -> (sum(*), ?, ?)
    # 3) Construct nested tensor
    #   (sum(*), ?, ?) -> (B, j0, ?, ?)
    #
    # If unsqueezing on the 0th dim becomes supported, we would unsqueeze
    # at step (4) and we would need to update this function to record how
    # many ones we unsqueezed.
    while t.shape[0] == 1:
        t = t.squeeze(0)
    return t


# unary pointwise
@register_njt2([
    torch.sin,
    torch.cos,
    torch.neg,
    torch.abs,
    torch.ones_like,
    torch.zeros_like,
    torch.empty_like,
    torch.nn.functional.relu,
    torch.nn.functional.silu,
])
def _(func, input, *, out=None, **kwargs):
    assert out is None
    return njt_like(input, func(input._values, **kwargs))

@register_njt2([torch.unbind])
def _(func, input, dim=0):
    if dim != 0:
        raise RuntimeError("unbind(): only supported for NestedTensor on dim=0")
    inp = input
    values = inp.values()
    offsets = inp.offsets()
    lengths = inp.lengths()

    if inp._ragged_idx != 1:
        raise RuntimeError(
            "unbind(): only supported for NestedTensor when jagged dimension is 1"
        )

    if lengths is None:
        return torch.split(values, offsets.diff().tolist())
    return [
        values[offsets[i] : (offsets[i] + lengths[i])] for i in range(lengths.shape[0])
    ]

# Simplifying assumption: we assume that the batch dim is always the left-most
# dim, and the ragged dim is always the second dim.
def _outer_to_inner_dim(ndim, dim):
    assert dim >= 0 and dim < ndim
    return 0 if dim < 2 else dim - 1

from torch.nested._internal.ops import _wrap_jagged_dim
from . import ops

@register_njt2(torch.transpose)
def transpose(func, input, dim0, dim1):
    return ops.transpose_int(NJT2, func, input=input, dim0=dim0, dim1=dim1)
