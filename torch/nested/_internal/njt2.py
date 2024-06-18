import torch
import itertools
from torch import Tensor
from typing import List, Optional, Tuple
from . import ops
import torch.utils._pytree as pytree

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

    def __eq__(self, other):
        return (self.a == other.a and self.b == other.b) or (self.a == other.b and self.b == other.a)

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

    @property
    def is_leaf(self):
        return self._values.is_leaf

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

    def reshape(self, *sizes):
        # TODO(rzou): this is not reshape
        return self.contiguous().view(*sizes)

    def contiguous(self, *args, **kwargs):
        return njt_like(self, self._values.contiguous(*args, **kwargs))


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

method_attrs = []
for attr in dir(torch.Tensor):
    if attr.startswith("_"):
        continue
    if attr in {"dtype", "numel", "device", "layout", "jagged"}:
        continue
    if attr in dir(NJT2):
        continue
    if hasattr(torch, attr):
        method_attrs.append(attr)

bind_method(method_attrs)

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
    torch.ops.aten.threshold_backward,
])
def _(func, input, other, *args, out=None, **kwargs):
    a = input
    b = other
    assert isinstance(a, NJT2) or isinstance(b, NJT2)
    assert out is None

    if not isinstance(other, torch.Tensor):
        # unary case
        return njt_like(input, func(input._values, other, *args, **kwargs))

    return ops.jagged_binary_pointwise(NJT2, func, input, other, *args, **kwargs)


import math
import torch.nn.functional as F

@register_njt2(torch.flatten)
def _(func, *args, **kwargs):
    return ops.jagged_torch_function(func, *args, **kwargs)

@register_njt2(torch.detach)
def _(func, *args, **kwargs):
    return torch.ops.aten.detach(*args, **kwargs)

@register_njt2(torch.chunk)
def _(func, inputs, chunks, dim=0):
    return ops.chunk_default(NJT2, func, input=inputs, chunks=chunks, dim=dim)

@register_njt2([
    torch.ops.aten.is_same_size
])
def _(func, *args, **kwargs):
    return ops.is_same_size_default(NJT2, func, *args, **kwargs)

@register_njt2(torch.split)
def _(func, tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, list):
        return ops.split_with_sizes_default(NJT2, torch.ops.aten.split_with_sizes, input=tensor, split_sizes=split_size_or_sections, dim=dim)
    return ops.split_tensor(NJT2, torch.ops.aten.split, input=tensor, split_size=split_size_or_sections, dim=dim)

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
@register_njt2(torch.nn.functional.scaled_dot_product_attention)
def _(func, *args, **kwargs):
    return ops.jagged_scaled_dot_product_attention(NJT2, *args, **kwargs)

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

@register_njt2(torch.autograd.grad)
def _(func, outputs, inputs, grad_outputs=None, **kwargs):
    if grad_outputs is not None:
        assert isinstance(outputs, NJT2) == isinstance(grad_outputs, NJT2)
    unwrap = lambda x: x._values if isinstance(x, NJT2) else x
    outputs_, inputs_, grad_outputs_ = pytree.tree_map(unwrap, (outputs, inputs, grad_outputs))
    results = torch.autograd.grad(outputs_, inputs_, grad_outputs_, **kwargs)
    results = tuple([njt_like(i, r) if isinstance(i, NJT2) else r for r, i in zip(results, inputs)])
    return results

# TODO(rzou): needs a good hard look
import dataclasses

@dataclasses.dataclass
class SDPAParams:
    query: Tensor
    key: Tensor
    value: Tensor
    attn_mask: Optional[Tensor]
    dropout: float
    is_causal: bool

@register_njt2(torch.nested._internal.sdpa._select_sdp_backend)
def _(func, query, key, value, attn_mask, dropout, is_causal):
    from torch.nested._internal.sdpa import (
        flash_sdp_enabled,
        mem_efficient_sdp_enabled,
        math_sdp_enabled,
        SDPBackend,
        _can_use_flash_sdpa_jagged,
        _can_use_efficient_sdpa_jagged,
        _can_use_math_sdpa_jagged,
    )

    def check_all_tensors_on_device(params):
        return params.query.device.type == "cuda"

    def check_tensor_shapes(params):
        query_dim = params.query.dim()
        if not (query_dim == params.key.dim() and query_dim == params.value.dim() and query_dim == 4):
            return False
        return True

    def check_for_attn_mask(params):
        return params.attn_mask is None

    def check_head_dim_size_flash(params):
        query_size_last = params.query.size(-1)
        key_size_last = params.key.size(-1)
        value_size_last = params.value.size(-1)
        return (query_size_last == key_size_last and query_size_last == value_size_last)

    def check_flash_causal_non_square_seqlens(params):
        if (params.is_causal and
                not params.query.is_nested() and not params.key.is_nested() and
                params.query.shape[-2] != params.key.shape[-2]):
            return False
        return True

    def check_dtypes_low_precision(params):
        query_dtype = params.query.dtype
        if not (query_dtype == params.key.dtype() and
                query_dtype == params.value.dtype() and
                query_dtype in {torch.half, torch.bfloat16}):
            return False
        return True

    def can_use_flash_attention(params):
        constraints = [
            # check_runtime_disabled_flash,
            check_all_tensors_on_device,
            check_tensor_shapes,
            check_for_attn_mask,
            check_head_dim_size_flash,
            # check_flash_attention_hardware_support,
            # check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89, # sm80 only for now
            check_flash_causal_non_square_seqlens,
            check_dtypes_low_precision,
        ]
        for constraint in constraints:
            if not constraint(params):
                return False
        return True


    def check_head_dim_size_mem_efficient(params):
        query_size_last = params.query.size(-1)
        value_size_last = params.value.size(-1)
        alignment = minimum_gemm_alignment(params)
        
        if not (query_size_last == params.key.sym_size(-1) and 
                query_size_last % alignment == 0 and query_size_last > 0 and 
                value_size_last % alignment == 0 and value_size_last > 0):
            return False
        
        return True

    def can_use_efficient_attention(params):
        constraints = [
            # check_runtime_disabled_mem_efficient,
            check_all_tensors_on_device,
            # check_mem_efficient_hardware_support,
            check_tensor_shapes,
            check_head_dim_size_mem_efficient,
        ]
        for constraint in constraints:
            if not constraint(params):
                return False
        return True

    def minimum_gemm_alignment(params):
        # dprops = torch.cuda.get_current_device_properties()
        is_half = (params.query.dtype == torch.float16) or (params.query.dtype == torch.bfloat16)
        # use_tc = use_tensor_cores(params, dprops, is_half)
        use_tc = True
        matmul_alignment_mn = 1
        # if dprops.major >= 8:
        if True:
            matmul_alignment_mn = 4
        bits_per_scalar = 16 if is_half else 32
        if use_tc:
            matmul_alignment_mn = max(matmul_alignment_mn, 128 // bits_per_scalar)
        return matmul_alignment_mn

    if (
        not flash_sdp_enabled()
        and not mem_efficient_sdp_enabled()
        and not math_sdp_enabled()
    ):
        return SDPBackend.ERROR

    ordering = (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    )

    params = SDPAParams(query, key, value, attn_mask, dropout, is_causal)

    for backend in ordering:
        if backend == SDPBackend.FLASH_ATTENTION:
            if can_use_flash_attention(params) and _can_use_flash_sdpa_jagged(params):
                return SDPBackend.FLASH_ATTENTION
        if backend == SDPBackend.EFFICIENT_ATTENTION:
            if can_use_efficient_attention(params) and _can_use_efficient_sdpa_jagged(
                params
            ):
                return SDPBackend.EFFICIENT_ATTENTION
        if backend == SDPBackend.MATH:
            if math_sdp_enabled() and _can_use_math_sdpa_jagged(params):
                return SDPBackend.MATH

    return SDPBackend.ERROR
