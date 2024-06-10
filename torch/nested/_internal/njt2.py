import torch
import itertools
from torch import Tensor
from typing import List, Optional, Tuple
from . import ops

NJT_OPS = {}


nid = itertools.count()

class NestedInt:
    def __init__(self, offsets):
        self._id = next(nid)
        self._offsets = offsets

    def __repr__(self):
        return f"j{self._id}"

    def __mul__(self, other):
        return NestedIntMul(self, other)

    def __rmul__(self, other):
        return NestedIntMul(other, self)


class NestedIntMul(NestedInt):
    def __init__(self, a, b):
        super().__init__(None)
        self.a = a
        self.b = b

    def __repr__(self):
        return f"{self.a}*{self.b}"

# TODO(rzou: weak key dictionary
NESTED_INT_CACHE = {}

class NJT2:
    def __repr__(self):
        return f"NJT2(shape={self.shape})"

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

    def numel(self):
        return self._values.numel()

    @property
    def _lengths(self):
        return None

    def requires_grad_(self, requires_grad):
        self._values.requires_grad_(requires_grad)
        return self

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def device(self):
        return self._values.device

    def is_contiguous(self, *args, **kwargs):
        return ops.is_contiguous_general(NJT2, torch.ops.aten.is_contiguous.default, input=self, *args, **kwargs)

    @property
    def shape(self):
        sizes = [self._offsets.shape[0] - 1, *self._values.shape]
        sizes[self._ragged_idx] = self._nested_int
        return tuple(sizes)

    def stride(self, dim=None):
        outer_stride = 1
        for s in self._values.stride():
            outer_stride *= s

        strides = [outer_stride * self._nested_int, *self._values.stride()]
        result = tuple(strides)
        if dim is None:
            return result
        return result[dim]

    def storage_offset(self):
        return self._values.storage_offset()

    @property
    def _size(self):
        return self.shape

    def detach(self):
        return torch.ops.aten.detach(self)

    @property
    def requires_grad(self):
        return self._values.requires_grad

    @property
    def layout(self):
        return torch.jagged

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def dim(self):
        return self._values.dim() + 1

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

    def view(self, *size):
        return torch.ops.aten.view(self, size)

    def expand(self, *sizes):
        return torch.ops.aten.expand(self, sizes)


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
    "squeeze",
    "unsqueeze",
    "clone",
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
def _(func, input, other, *, out=None, **kwargs):
    a = input
    b = other
    assert isinstance(a, NJT2) or isinstance(b, NJT2)
    assert out is None

    if not isinstance(other, torch.Tensor):
        # unary case
        return njt_like(input, func(input._values, other, **kwargs))

    return ops.jagged_binary_pointwise(NJT2, func, input, other, **kwargs)


import math
import torch.nn.functional as F

@register_njt2(torch.chunk)
def _(func, inputs, chunks, dim=0):
    return ops.chunk_default(NJT2, func, inputs=inputs, chunks=chunks, dim=dim)

# unary pointwise
@register_njt2([
    torch.sin,
    torch.cos,
    torch.neg,
    torch.abs,
    torch.ones_like,
    torch.zeros_like,
    torch.empty_like,
    torch.randn_like,
    torch.rand_like,
    torch.clone,
    torch.nn.functional.relu,
    torch.nn.functional.silu,
    torch.ops.aten._to_copy,
    torch.ops.aten.detach,
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



@register_njt2(torch.transpose)
def transpose(func, input, dim0, dim1):
    return ops.transpose_int(NJT2, func, input=input, dim0=dim0, dim1=dim1)

@register_njt2(torch.squeeze)
def _(func, input, dim=None):
    return ops.squeeze_dim(NJT2, func, input=input, dim=dim)

@register_njt2(torch.unsqueeze)
def _(func, input, dim):
    return ops.unsqueeze_default(NJT2, func, input=input, dim=dim)

@register_njt2(torch.nn.functional.linear)
def _(func, input, weight, bias=None):
    return ops.linear_default(NJT2, func, input=input, weight=weight, bias=bias)

@register_njt2(torch.sum)
def _(func, input, dim=None, keepdim=False, *, dtype=None):
    if dim is None:
        dim = list(range(input.dim()))
    return ops.sum_dim_IntList(NJT2, func, input=input, dim=dim, keepdim=keepdim, dtype=dtype)

# TODO(rzou): not sure the eps
@register_njt2(torch.nn.functional.layer_norm)
def _(func, input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return ops.native_layer_norm_default(NJT2, torch.ops.aten.native_layer_norm.default, input=input, normalized_shape=normalized_shape, weight=weight, bias=bias, eps=eps)[0]

# TODO(rzou): segfault
# @register_njt2(torch.nn.functional.scaled_dot_product_attention)
# def _(func, *args, **kwargs):
#     return ops.jagged_scaled_dot_product_attention(NJT2, *args, **kwargs)

@register_njt2([
    torch.ops.aten.view,
    torch.ops.aten._unsafe_view,
])
def _(func, self, size):
    return ops.view_default(NJT2, func, input=self, size=size)

@register_njt2(torch.ops.aten.expand)
def _(func, self, size, implicit=False):
    return ops.expand_default(NJT2, func, input=self, size=size, implicit=implicit)

@register_njt2(torch.ops.aten.is_non_overlapping_and_dense.default)
def _(func, self, *args, **kwargs):
    return func(self._values, *args, **kwargs)

@register_njt2(torch.ops.aten.sym_size.default)
def _(func, self, *args, **kwargs):
    return self.size(*args, **kwargs)

@register_njt2(torch.ops.aten.dim.default)
def _(func, self, *args, **kwargs):
    return self.dim()

@register_njt2([torch.ops.aten.numel.default, torch.ops.aten.sym_numel.default])
def _(func, self, *args, **kwargs):
    return self.numel()

@register_njt2([torch.ops.aten.stride.default, torch.ops.aten.sym_stride.default])
def _(func, self, *args, **kwargs):
    return self.stride(*args, **kwargs)

@register_njt2([torch.ops.aten.sym_storage_offset.default])
def _(func, self, *args, **kwargs):
    return self.storage_offset()
