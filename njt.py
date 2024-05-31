import torch
import itertools
from torch import Tensor

NJT_OPS = {}


nid = itertools.count()

class NestedInt:
    def __init__(self, njt):
        self._id = next(nid)
        self._njt = njt

    def __repr__(self):
        return f"j{self._id}"

class NJT:
    def __init__(self, values, offsets, ragged_idx=1):
        self._values = values
        self._offsets = offsets
        assert self._values.device == self._offsets.device
        self._nested_int = NestedInt(self)
        self._ragged_idx = ragged_idx

    @property
    def __class__(self):
        # TODO(rzou): is this bad?
        return torch.Tensor

    def values(self):
        return self._values

    def offsets(self):
        return self._offsets

    def lengths(self):
        return None

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
    def requires_grad(self):
        return self._values.requires_grad

    @property
    def layout(self):
        return torch.jagged

    def size(self):
        return self.shape

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
            return NJT(self._values.to(*args, **kwargs), self._offsets.to(kwargs["device"]))
        else:
            return NJT(self._values.to(*args, **kwargs), self._offsets)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __add__(self, other):
        return torch.add(self, other)

    def __sub__(self, other):
        return torch.sub(self, other)

    @property
    def grad(self):
        return NJT(self._values.grad, self._offsets)

    def backward(self, grad_output=None):
        if grad_output is not None and isinstance(grad_output, torch.Tensor):
            assert same_offsets(grad_output, self)
            self._values.backward(grad_output._values)
        else:
            self._values.backward(grad_output)


# Not going to work for in-place
def bind_method(api):
    if isinstance(api, str):
        func = getattr(torch, api)
        def method(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        setattr(NJT, api, method)
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
])

def as_nested_tensor(ts, dtype=None, device=None, layout=None):
    is_tensor_list = isinstance(ts, (list, tuple)) and all(isinstance(t, Tensor) for t in ts)
    if not isinstance(ts, Tensor) and not is_tensor_list:
        raise TypeError(
            "as_nested_tensor(): Expected first argument to be a tensor or a list / tuple of tensors "
        )
    # convert tuple -> list if needed
    if is_tensor_list and not isinstance(ts, list):
        ts = list(ts)

    if isinstance(ts, Tensor) and ts.dim() < 2:
        raise RuntimeError("as_nested_tensor(): Expected tensor argument to have dim() > 1")

    if isinstance(ts, Tensor) and ts.is_nested:
        if layout == ts.layout:
            # return input directly or input copied to device / dtype
            return ts.to(device=device, dtype=dtype)
        else:
            # TODO: Just use nt.to(layout=layout) when it exists.
            raise RuntimeError(
                "as_nested_tensor(): Converting between nested tensor layouts is not supported")
    if layout is None:
        layout = torch.strided

    if layout != torch.jagged:
        raise RuntimeError(f"Specified layout is unsupported for nested tensors: {layout}")

    if isinstance(ts, torch.Tensor):
        values = ts.contiguous().flatten(0, 1).to(device=device, dtype=dtype)
        batch_size = ts.shape[0]
        seq_len = ts.shape[1]
        offsets = torch.arange(0, batch_size * seq_len + 1, seq_len,
                               device=device, dtype=torch.int64)
        return NJT(values, offsets)
    else:
        assert isinstance(ts, list)
        tensors = ts
        if not len(set(t.dtype for t in tensors)) == 1:  # noqa: C401
            raise RuntimeError(
                "When constructing a nested tensor, all tensors in list must have the same dtype"
            )
        if not len(set(t.device for t in tensors)) == 1:  # noqa: C401
            raise RuntimeError(
                "When constructing a nested tensor, all tensors in list must be on the same device"
            )

        # Check that the NT is representable by the jagged layout.
        # Jagged layout represents (B, *, D_0, D_1, ..., D_N), where the only
        # raggedness allowed is for the single dim immediately adjacent to the batch dim.
        sizes = [t.shape for t in tensors]
        non_first_sizes = [s[1:] for s in sizes]
        at_most_first_ragged = all(s == non_first_sizes[0] for s in non_first_sizes)
        if not at_most_first_ragged:
            raise RuntimeError(
                "Cannot represent given tensor list as a nested tensor with the jagged layout. "
                "Note that the jagged layout only represents shapes of the form "
                "(B, *, D_0, D_1, ..., D_N), with only * allowed to be ragged."
            )

        # Set properties appropriately.
        values = torch.cat(tensors, dim=0)
        to_kwargs = {}
        if device is not None:
            to_kwargs["device"] = device
        if dtype is not None:
            to_kwargs["dtype"] = dtype
        values = values.to(**to_kwargs)

        offsets = None

        # Calculate jagged offsets if not provided.
        if offsets is None:
            # Jagged layout specifies that offsets are stored as int64 on the same device as values.
            # TODO: An alternative way to construct offsets is to use F.pad. This avoids creating
            # an extra leaf tensor during the forward, potentially resolving compatibility issues.
            offsets = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=values.device),
                    torch.tensor([s[0] for s in sizes], device=values.device).cumsum(dim=0),
                ]
            )

        return NJT(values, offsets)

print("!!!! Monkeypatching torch.nested.as_nested_tensor !!!!")
torch.nested.as_nested_tensor = as_nested_tensor

def register_njt(func):
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

def same_offsets(a, b):
    # TODO(rzou): try jeffrey's union find thing
    return a._offsets is b._offsets

# binary pointwise
@register_njt([
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
])
def _(func, input, other, *, out=None, **kwargs):
    assert out is None
    if isinstance(other, torch.Tensor):
        if not same_offsets(input, other):
            raise RuntimeError("cannot call binary pointwise function .* with inputs of shapes")
        return NJT(func(input._values, other._values, **kwargs), input._offsets)
    else:
        return NJT(func(input._values, other, **kwargs), input._offsets)

# unary pointwise
@register_njt([
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
    return NJT(func(input._values, **kwargs), input._offsets)

@register_njt([torch.unbind])
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

@register_njt(torch.transpose)
def transpose(func, input, dim0, dim1):
    from torch._prims_common import canonicalize_dims

    inp = input
    dim0, dim1 = canonicalize_dims(inp.dim(), (dim0, dim1))

    if inp._lengths is not None:
        raise ValueError(
            "transpose(): not supported on jagged layout nested tensor with holes"
        )

    # To support the SDPA API, inputs need to have the ragged idx transposed to dim 2
    # instead of 1, although the internal Flash and mem-effn implementations will
    # use the inputs with raggedness in dim 1.
    if dim0 == inp._ragged_idx or dim1 == inp._ragged_idx:
        if dim0 == 0 or dim1 == 0:
            raise ValueError(
                "Transpose is not supported on the batch dimension for jagged NT"
            )
        if dim0 == inp._ragged_idx:
            to_dim = dim1
        else:
            to_dim = dim0
        return NJT(
            inp.values().transpose(
                _outer_to_inner_dim(len(inp._size), dim0),
                _outer_to_inner_dim(len(inp._size), dim1),
            ),
            inp._offsets
            ragged_idx=to_dim
        )

    new_kwargs["dim0"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim0"], "transpose")
    new_kwargs["dim1"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim1"], "transpose")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))

