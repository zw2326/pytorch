# Owner(s): ["module: nestedtensor"]

import io
import itertools
import sys
from typing import Optional, Tuple
import unittest
from functools import partial
import math

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.testing._internal.common_cuda import (
    SM70OrLater, SM80OrLater, PLATFORM_SUPPORTS_FUSED_ATTENTION,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    skipCUDAIf,
    skipCUDAIfRocm,
    skipMeta,
    PYTORCH_CUDA_MEMCHECK,
)
from torch.testing._internal.common_dtype import floating_types_and_half
from torch.testing._internal.common_utils import (
    decorateIf,
    freeze_rng_state,
    gradcheck,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfSlowGradcheckEnv,
    skipIfTorchDynamo,
    markDynamoStrictTest,
    xfailIfTorchDynamo,
    subtest,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.nested._internal.nested_tensor import (
    buffer_from_jagged,
    jagged_from_list2,
    NestedTensor,
    nested_view_from_values_offsets,
    ViewNestedFromBuffer,
)

from torch.nested._internal import njt2

# Tests are ported from pytorch/nestedtensor.
# This makes porting as_nested_tensor2 easier in the future.


def _iter_constructors():
    # yield as_nested_tensor2
    yield torch.nested.nested_tensor

# Helper function to generate a pair of random nested tensors
# one is contiguous, the other is not, but they appear to have same entries
# an output nested tensor consists of
# * `len(ragged_sizes)` matrices
# * matrices[i].shape == (20, ragged_sizes[i])


def random_nt_noncontiguous_pair(ragged_sizes, device="cpu", dtype=torch.float16):
    xs = []
    for size in ragged_sizes:
        xs.append(torch.randn((size, 20), device=device, dtype=dtype))
    # contiguous nested tensor
    ys = []
    for x in xs:
        ys.append(x.transpose(-1, -2))
    nt_contiguous = torch.nested.nested_tensor2(ys)
    # noncontiguous nested tensor
    n = len(ragged_sizes)
    nt_noncontiguous = torch.nested.nested_tensor2(xs).transpose(-1, -2)
    return nt_contiguous, nt_noncontiguous

# Helper functions to pad a noncontiguous nested tensor
# can be replaced once to_padded_tensor supports noncontiguous memory


def noncontiguous_to_padded_tensor(input, shape=None):
    tensors = input.unbind()
    ntensors = len(tensors)
    assert ntensors > 0
    if shape is None:
        shape = []
        for size in tensors[0].shape:
            shape.append(size)
        for i in range(1, ntensors):
            new_shape = tensors[i].shape
            for j in range(len(shape)):
                shape[j] = max(shape[j], new_shape[j])
        shape = [ntensors] + shape
    result = tensors[0].new_zeros(shape)
    for itensor in range(ntensors):
        tensor = tensors[itensor]
        view = result[itensor]
        for idim in range(tensor.dim()):
            view = view.narrow(idim, 0, tensor.size(idim))
        view.copy_(tensor)
    return result

# Found in torch/testing/_comparison.py
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}

def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    deviation = torch.abs(deviation / true_value)
    # Fill in the nans with the default rtol
    torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    return deviation.max().item()


def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    atol = torch.abs(deviation).max().item()
    return atol


def get_tolerances(
    true_value: torch.Tensor,
    computed_value: torch.Tensor,
    fudge_factor: Optional[float] = None,
) -> Tuple[float, float]:
    """Returns the absolute and relative tolerances for comparing two tensors."""
    fudge_factor = fudge_factor if fudge_factor is not None else 1.0
    atol = get_atol(true_value, computed_value)
    rtol = get_rtol(true_value, computed_value)

    atol = fudge_factor * max(atol, default_atol[computed_value.dtype])
    rtol = fudge_factor * max(rtol, default_rtol[computed_value.dtype])
    # torch.isclose() has weird behavior around see:
    # https://github.com/pytorch/pytorch/issues/102400
    if rtol > 1e30:
        rtol = default_rtol[computed_value.dtype]
    return atol, rtol

# We can probably parametrizing existing tests instead of having a separate

# Helper function to generate a random nested tensor


def random_nt(device, dtype, num_tensors, max_dims, min_dims=None, layout=torch.strided, require_non_empty=True):
    if min_dims is None:
        min_dims = tuple([0] * len(max_dims))

    assert len(max_dims) == len(min_dims)
    for min_dim, max_dim in zip(min_dims, max_dims):
        assert max_dim > min_dim, "random_nt: max_dim must be greater than min_dim"
        assert min_dim >= 0, "random_nt: min_dim must be non-negative"
        if require_non_empty:
            assert not (min_dim == 0 and max_dim == 1), (
                "random_nt: zero cannot be the only possible value if require_non_empty is True"
            )

    if require_non_empty:
        # Select a random idx that will be required to be non-empty
        non_zero_idx = torch.randint(low=0, high=num_tensors, size=(1,)).item()

    ts1 = []
    for i, _ in enumerate(range(num_tensors)):
        tensor_dims = []
        for min_dim, max_dim in zip(min_dims, max_dims):
            new_min_dim = min_dim
            if require_non_empty and i == non_zero_idx and min_dim == 0:
                new_min_dim = 1
            tensor_dims.append(torch.randint(low=new_min_dim, high=max_dim, size=(1,)).item())
        t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
        ts1.append(t1)

    return torch.nested.nested_tensor2(ts1, device=device, dtype=dtype, layout=layout)


# Alternate approach to generating a random NT.
# dims should be something like [5, None, 10], with None indicating that a
# random ragged structure should be used
def random_nt_from_dims(dims, device=None, dtype=None, layout=torch.strided, requires_grad=False):
    sizes = [
        [d if d is not None else torch.randint(2, 10, size=(1,)).item() for d in dims[1:]]
        for d in range(dims[0])
    ]
    result = torch.nested.nested_tensor2([
        torch.randn(*size) for size in sizes
    ], device=device, dtype=dtype, layout=layout, requires_grad=requires_grad)
    assert result.dim() == len(dims)
    return result


# Creates an NT matching another NT's number of components and
# shape / ragged structure for all dims specified to be -1.
def random_nt_from_similar(other, dims=None):
    if dims is None:
        return torch.randn_like(other)
    assert len(dims) == other.dim()
    assert dims[0] == -1 or dims[0] == other.size(0)

    ret_sizes = []
    for t in other.unbind():
        other_size = t.shape
        ret_size = []
        for i, d in enumerate(dims[1:]):
            if d == -1:
                ret_size.append(other_size[i])
            else:
                ret_size.append(d)
        ret_sizes.append(ret_size)

    return torch.nested.nested_tensor2([
        torch.randn(*size) for size in ret_sizes
    ], device=other.device)


# makes naming nice for tests that parametrize over layout.
def layout_name(layout):
    # e.g. "torch.jagged" -> "jagged"
    return layout.__repr__().split(".")[-1]


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_dense_to_nested_tensor(values):
    offsets = torch.arange(
        0, values.shape[0] * values.shape[1] + 1, values.shape[1], device=values.device
    )
    metadata_cache = {"max_seqlen": values.shape[1], "min_seqlen": 1}
    nt = ViewNestedFromBuffer.apply(
        values.view(-1, values.shape[-1]), offsets, metadata_cache
    )
    return nt


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_jagged_to_nested_tensor(
    values: torch.Tensor, offsets: torch.Tensor, max_length: int
) -> torch.Tensor:
    metadata_cache = {"max_seqlen": max_length, "min_seqlen": 1}
    nt = ViewNestedFromBuffer.apply(values, offsets, metadata_cache)
    return nt


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_nt_to_jagged(nt):
    return buffer_from_jagged(nt)


# We can probably parametrizing existing tests instead of having a separate
# test class as we begin to support more ops. Also maybe rewrite with OpInfos.
@markDynamoStrictTest
class TestNestedTensorSubclass(TestCase):
    # TODO: consolidate with the below
    def _get_list_for_jagged_tensor(self, nested_size, device, requires_grad=True):
        Ds = nested_size[1:]
        out = []
        for s in nested_size[0]:
            out.append(
                torch.randn(s, *Ds, requires_grad=requires_grad, device=device, dtype=torch.float64)
            )
        return out

    def _get_example_tensor_lists(self, include_list_of_lists=True, include_requires_grad=True):

        def _make_tensor(*shape, include_requires_grad=include_requires_grad, requires_grad=True):
            return torch.randn(
                *shape,
                requires_grad=(requires_grad if include_requires_grad else False)
            )

        # Purposefully introduce mixed requires_grad settings for the components
        # when include_requires_grad=True.
        example_lists = [
            # (B, *, D) with B=4
            [
                _make_tensor(2, 5),
                _make_tensor(3, 5, requires_grad=False),
                _make_tensor(4, 5, requires_grad=False),
                _make_tensor(6, 5)
            ],
            # (B, *, D_0, D_1) with B=5
            [
                _make_tensor(2, 5, 6),
                _make_tensor(3, 5, 6),
                _make_tensor(4, 5, 6, requires_grad=False),
                _make_tensor(5, 5, 6),
                _make_tensor(6, 5, 6),
            ],
        ]

        if include_list_of_lists:
            example_lists.append(
                # (B, *, D) with B=3 in list form
                [
                    _make_tensor(2, 5, requires_grad=False).tolist(),
                    _make_tensor(3, 5).tolist(),
                    _make_tensor(4, 5).tolist(),
                ])

        return example_lists

    def test_tensor_attributes(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)
        nt = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)
        _offsets = nt.offsets()

        for op in (
            torch.ops.aten.is_non_overlapping_and_dense.default,
            torch.ops.aten.sym_size.default,
            torch.ops.aten.dim.default,
            torch.ops.aten.numel.default,
            torch.ops.aten.sym_numel.default,
            torch.ops.aten.sym_stride.default,
            torch.ops.aten.sym_storage_offset.default,
        ):
            op(nt)

        # with self.assertRaisesRegex(RuntimeError,
        #                             "directly calling torch.ops.aten.size"):
        #     torch.ops.aten.size.default(nt)

        # TODO(rzou): this needs a monkey-patch too
        # nested_int = torch.nested._internal.nested_tensor.get_tensor_symint(_offsets, coeff=1)
        # self.assertEqual(nt.size(), (3, nested_int, 3))
        # self.assertEqual(nt.shape, (3, nested_int, 3))
        self.assertEqual(nt.dim(), 3)
        self.assertEqual(nt.numel(), 27)

    def test_linear(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)
        weight = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, weight):
            nt = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)
            out = torch.nn.functional.linear(nt, weight)
            return out.values()

        gradcheck(grad_test_func, inputs=(a, b, c, weight), check_batched_grad=False)

    def test_unary_pointwise(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)
            out = torch.nn.functional.silu(nt.sin().cos())
            return out.values()

        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)

    def test_unary_pointwise_transposed_inputs(self, device):
        a, b, c = (
            torch.randn(i + 2, 5, requires_grad=True, dtype=torch.float64, device=device) for i in range(3)
        )

        nt = torch.nested.nested_tensor2([a.detach(), b.detach(), c.detach()], layout=torch.jagged)
        nt_t = nt.transpose(1, 2)
        self.assertFalse(nt_t.is_contiguous())
        out = torch.nn.functional.silu(nt_t.sin().cos())
        self.assertEqual(out.is_contiguous(), torch.nn.functional.silu(b.transpose(-1, -2).sin().cos()).is_contiguous())

        self.assertEqual(nt_t.shape, out.shape)

        a, b, c = (
            torch.randn(i + 2, 5, requires_grad=True, dtype=torch.float64, device=device) for i in range(3)
        )

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)
            nt_t = nt.transpose(1, 2)
            out = torch.nn.functional.silu(nt_t.sin().cos())
            return out.values()

        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)


    def test_binary_pointwise(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        # Incorrect usage: shape check will fail if the offsets tensor are not
        #                  the same exact tensor object
        nt1 = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)
        nt2 = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)

        self.assertRaises(
            RuntimeError,
            lambda: nt1 * nt2)

        # Correct usage: chain the calls using the same offsets tensor object
        def grad_test_func(a, b, c):
            nt1 = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)
            nt2 = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)
            nt2._offsets = nt1._offsets
            nt2._nested_int = nt1._nested_int
            # TODO: Switch to public API that takes in (values, offsets) once it exists
            # nt2, offsets = jagged_from_list2([a, b, c], nt1.offsets())
            out = nt1 * nt2
            return out.values()

        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)

    def test_binary_pointwise_transposed(self, device):
        a, b, c = (
            torch.randn(i + 2, 5, dtype=torch.float64, device=device) for i in range(3)
        )

        nt1, offsets = jagged_from_list2([a, b, c], None)
        nt2, offsets = jagged_from_list2([a, b, c], offsets)

        nt1_t = nt1.transpose(1, 2)
        nt2_t = nt2.transpose(1, 2)

        # out = nt1_t * nt2_t
        # self.assertFalse(nt1_t.is_contiguous())
        # self.assertEqual(out.is_contiguous(), (b.transpose(-1, -2) * b.transpose(-1, -2)).is_contiguous())
        # self.assertEqual(out.shape, nt1_t.shape)

        self.assertRaises(
            RuntimeError,
            lambda: nt1 * nt2_t,
        )

        a, b, c = (
            torch.randn(i + 2, 5, requires_grad=True, dtype=torch.float64, device=device) for i in range(3)
        )

        # Correct usage: chain the calls using the same offsets tensor object
        def grad_test_func(a, b, c):
            nt1, offsets = jagged_from_list2([a, b, c], None)
            nt2, offsets = jagged_from_list2([a, b, c], offsets)
            nt1_t = nt1.transpose(1, 2)
            nt2_t = nt2.transpose(1, 2)
            out = nt1_t * nt2_t
            return out.values()

        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)

    def test_split(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        nt = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)
        out = torch.split(nt, 2, -1)
        self.assertEqual(len(out), 2)
        self.assertEqual(
            out[0],
            torch.nested.as_nested_tensor2([a[:, 0:2], b[:, 0:2], c[:, 0:2]], layout=torch.jagged)
        )
        self.assertEqual(
            out[1],
            torch.nested.as_nested_tensor2([a[:, 2:], b[:, 2:], c[:, 2:]], layout=torch.jagged)
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"split\(\): not supported for NestedTensor on dim=1",
        ):
            torch.split(nt, 2, 1)

    def test_split_with_sizes(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        nt = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)
        out = torch.split(nt, [1, 2], -1)
        self.assertEqual(len(out), 2)
        self.assertEqual(
            out[0],
            torch.nested.as_nested_tensor2([a[:, 0:1], b[:, 0:1], c[:, 0:1]], layout=torch.jagged)
        )
        self.assertEqual(
            out[1],
            torch.nested.as_nested_tensor2([a[:, 1:], b[:, 1:], c[:, 1:]], layout=torch.jagged)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"split_with_sizes\(\): not supported for NestedTensor on dim=1",
        ):
            torch.split(nt, [1, 2], 1)

    def test_views_inherit_ragged_dim(self, device):
        # view
        nt = random_nt_from_dims(
            [4, None, 8, 10], device=device, dtype=torch.float32, layout=torch.jagged)
        # inherit ragged dim via -1
        view = nt.view(4, -1, 80)
        self.assertEqual(nt.shape[1], view.shape[1])
        # inherit batch and ragged dims via -1
        view2 = nt.view(-1, -1, 80)
        self.assertEqual(nt.shape[:2], view2.shape[:2])

        # expand
        nt = random_nt_from_dims(
            [3, None, 1], device=device, dtype=torch.float32, layout=torch.jagged)
        # inherit batch and ragged dims via -1
        view = nt.expand(-1, -1, 5)
        self.assertEqual(nt.shape[:2], view.shape[:2])

    def test_view_ragged_idx_not_one(self, device):
        nt = random_nt_from_dims([2, None, 20], device=device, dtype=torch.float32, layout=torch.jagged)

        tt = nt.transpose(1, 2)

        view_transposed = nt.transpose(1, 2).view(2, 20, nt.size(1))
        self.assertEqual((2, 20, nt.size(1)), (view_transposed.size()))
        # self.assertEqual(view_transposed._base, nt._base)

    def test_unsafe_view(self, device):
        nt = random_nt_from_dims([4, None, 8, 10], device=device, dtype=torch.float32, layout=torch.jagged)
        # basic view
        view1 = torch.ops.aten._unsafe_view(nt, (4, -1, 80))
        self.assertEqual((4, nt.size(1), 80), tuple(view1.size()))
        # _unsafe_view differs from view in that the view information is not tracked
        # self.assertTrue(view1._base is None)

        # test an unsafe_view when ragged_idx != 1, currently only supports identity view
        nt_t = nt.transpose(1, 2)
        view2 = torch.ops.aten._unsafe_view(nt_t, (4, 8, nt.size(1), 10))
        self.assertEqual((4, 8, nt.size(1), 10), tuple(view2.size()))
        # self.assertTrue(view2._base is None)

    @parametrize("requires_grad", [False, True])
    def test_reshape_decomp(self, device, requires_grad):
        # contiguous NT should result in view.
        nt = random_nt_from_dims(
            [3, None, 10],
            device=device,
            dtype=torch.float32,
            layout=torch.jagged,
        ).detach().requires_grad_(requires_grad)
        view = nt.reshape(-1, -1, 5, 2)
        self.assertEqual(view.shape[:2], nt.shape[:2])
        # self.assertTrue(view._is_view() and view._base is nt)
        # make sure gradients flow back
        if requires_grad:
            view.backward(torch.ones_like(view))
            self.assertEqual(nt.grad, torch.ones_like(nt))

        # non-contiguous NT should result in contiguous copy
        nt = random_nt_from_dims(
            [3, None, 5, 2],
            device=device,
            dtype=torch.float32,
            layout=torch.jagged,
            requires_grad=requires_grad
        )
        nt_noncontig = nt.transpose(-1, -2)
        self.assertFalse(nt_noncontig.is_contiguous())
        copy = nt_noncontig.reshape(-1, -1, 10)
        self.assertTrue(copy.is_contiguous())
        self.assertEqual(copy.shape[:2], nt.shape[:2])
        # make sure gradients flow back
        if requires_grad:
            copy.backward(torch.ones_like(copy))
            self.assertEqual(nt.grad, torch.ones_like(nt))

    def test_flatten_decomp(self, device):
        nt = random_nt_from_dims(
            [3, None, 5, 2], device=device, dtype=torch.float32, layout=torch.jagged)
        flattened = nt.flatten(-2, -1)
        self.assertEqual(flattened.shape, nt.view(3, -1, 10).shape)

        nt = random_nt_from_dims(
            [3, None, 5, 2, 6], device=device, dtype=torch.float32, layout=torch.jagged)
        flattened = nt.flatten(-3, -2)
        self.assertEqual(flattened.shape, nt.view(3, -1, 10, 6).shape)

    def test_chunk(self, device):
        # normal case
        D = 30
        B = 8
        nt = random_nt_from_dims([B, None, D], device=device, dtype=torch.float32, layout=torch.jagged)
        NUM_CHUNKS = 3
        chunks = nt.chunk(NUM_CHUNKS, dim=-1)
        self.assertEqual(len(chunks), NUM_CHUNKS)
        for i in range(NUM_CHUNKS):
            self.assertEqual(chunks[i].shape[-1], D // NUM_CHUNKS)

        # chunk on batch dim
        chunks = nt.chunk(NUM_CHUNKS, dim=0)
        self.assertEqual(len(chunks), NUM_CHUNKS)
        chunk_size = math.ceil(B / NUM_CHUNKS)
        for i in range(NUM_CHUNKS):
            if i < NUM_CHUNKS - 1:
                self.assertEqual(chunks[i].shape[0], chunk_size)
            else:
                self.assertEqual(chunks[i].shape[0], B - chunk_size * (NUM_CHUNKS - 1))
            offsets_expected = nt._offsets[i * chunk_size + 1 : (i + 1) * chunk_size + 1] - nt._offsets[i * chunk_size]
            self.assertEqual(chunks[i]._offsets[1:], offsets_expected)
        self.assertEqual(nt._values, torch.cat([x._values for x in chunks], dim=0))

        # chunk on ragged dim not supported
        with self.assertRaisesRegex(RuntimeError, "chunk.* not supported for NestedTensor on dim=1"):
            nt.chunk(2, dim=1)

    def test_squeeze(self, device):
        B = 4
        D = 6
        # squeeze middle dim
        nt = random_nt_from_dims(
            [B, None, 1, D], device=device, dtype=torch.float32, layout=torch.jagged)
        j0 = nt.shape[1]

        for dim_arg in [-2, 2]:
            out = nt.squeeze(dim_arg)
            self.assertEqual(out.shape, (B, j0, D))
            self.assertEqual(out.unsqueeze(-2), nt)

        # squeeze last dim
        nt = random_nt_from_dims(
            [B, None, 1], device=device, dtype=torch.float32, layout=torch.jagged)
        j1 = nt.shape[1]

        for dim_arg in [-1, 2]:
            out = nt.squeeze(dim_arg)
            self.assertEqual(out.shape, (B, j1))
            self.assertEqual(out.unsqueeze(-1), nt)

        # squeeze on batch dim not supported
        with self.assertRaisesRegex(
                RuntimeError, "squeeze.* not supported for NestedTensor on dim=0"):
            nt.squeeze(0)

        # squeeze on ragged dim not supported
        with self.assertRaisesRegex(
                RuntimeError, "squeeze.* not supported for NestedTensor on dim=1"):
            nt.squeeze(1)

    def test_binary_pointwise_broadcasting(self, device):
        # (B, j0, 3, 4)
        ts = self._get_list_for_jagged_tensor(((2, 3, 4), 3, 4), device, requires_grad=True)
        # (B, j0, ?, ?) + (?) -> (B, j0, ?, ?)
        # (B, j0, ?, ?) + (?, ?) -> (B, j0, ?, ?)
        # (B, j0, ?, ?) + (1, ?, ?) -> (B, j0, ?, ?)
        # Unsupported: (B, j0, ?, ?) + (1, 1, 1, ?, ?) -> (1, B, j0, ?, ?)
        t_sizes = (
            (4,),
            (1, 4),
            (3, 1),
            (1, 3, 1),
            (1, 1, 1, 4),
            # (1, 1, 1, 1, 4), (unsupported today)
        )

        def grad_test_func(t, *ts):
            nt = torch.nested.as_nested_tensor2(list(ts), layout=torch.jagged)
            out = nt + t
            return out.values()

        for t_size in t_sizes:
            t = torch.rand(t_size, requires_grad=True, device=device, dtype=torch.float64)
            gradcheck(grad_test_func, inputs=(t, *ts), check_batched_grad=False)

    def test_threshold_backward(self, device):
        ts1 = self._get_list_for_jagged_tensor(((2, 3, 4), 16), device=device, requires_grad=False)
        ts2 = self._get_list_for_jagged_tensor(((2, 3, 4), 16), device=device, requires_grad=False)

        nt1, offsets = jagged_from_list2(ts1, None)
        nt2, offsets = jagged_from_list2(ts2, offsets)
        buf1 = nt1.values().detach().clone()
        buf2 = nt2.values().detach().clone()

        res_nt = torch.ops.aten.threshold_backward(nt1, nt2, 0.0)
        res_dense = torch.ops.aten.threshold_backward(buf1, buf2, 0.0)

        self.assertEqual(res_dense, res_nt.values())


    @parametrize("keepdim", [False, True])
    def test_sum_int_DimList(self, device, keepdim):
        # (B, j0, 3, 4)
        ts = self._get_list_for_jagged_tensor(((2, 3, 4), 3, 4), device=device, requires_grad=True)

        # Check shape correctness
        reduce_dims = (
            # dims, expected shape, expected keepdim shape
            # j0 is represented as None
            ((0, 1), (3, 4), (1, 1, 3, 4)),
            ((1, 2), None, None),
            ((2, 3), (3, None), (3, None, 1, 1)),
            ((0, 1, 3), (3,), (1, 1, 3, 1)),
            ((0, 1, 2), (4,), (1, 1, 1, 4)),
            ((0, 1, 2, 3), tuple(), (1, 1, 1, 1)),
        )
        for rd, ref_shape_no_keepdim, ref_shape_keepdim in reduce_dims:
            if (0 in rd) ^ (1 in rd):
                with self.assertRaisesRegex(
                        RuntimeError,
                        "applying over the ragged dimension, but not the batch dimension"):
                    nt = torch.nested.as_nested_tensor2(ts, layout=torch.jagged)
                    out = torch.sum(nt, dim=rd, keepdim=keepdim)
                continue

            nt = torch.nested.as_nested_tensor2(ts, layout=torch.jagged)
            out = torch.sum(nt, dim=rd, keepdim=keepdim)
            ref_shape = ref_shape_keepdim if keepdim else ref_shape_no_keepdim
            self.assertEqual(len(out.shape), len(ref_shape))
            for o, r in zip(out.shape, ref_shape):
                if r is not None:
                    self.assertEqual(o, r)
                else:
                    self.assertFalse(isinstance(o, int))
                    # self.assertTrue(isinstance(o, torch.SymInt))

        # Check values correctness
        # raggedness not reduced
        nt = torch.nested.as_nested_tensor2(ts, layout=torch.jagged)
        out = torch.sum(nt, dim=(2, 3), keepdim=keepdim)
        out_ref = torch.sum(nt.values(), dim=(1, 2))
        self.assertIsInstance(out, torch.nested._internal.njt2.NJT2)
        # flatten to avoid having to replicate unsqueeze logic depending on keepdim
        self.assertTrue(torch.allclose(out.values().view(-1), out_ref.view(-1)))

        # raggedness reduced away
        nt = torch.nested.as_nested_tensor2(ts, layout=torch.jagged)
        out = torch.sum(nt, dim=(0, 1), keepdim=keepdim)
        out_ref = torch.sum(nt.values(), dim=(0,))
        self.assertNotIsInstance(out, torch.nested._internal.njt2.NJT2)
        self.assertTrue(torch.allclose(out, out_ref))



    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("requires_grad", [False, True])
    @parametrize("weights_only", [False, True])
    def test_serialization(self, device, dtype, requires_grad, weights_only):

        def compare_metadata(nt1, nt2):
            self.assertEqual(nt1._nested_tensor_size(), nt2._nested_tensor_size())
            self.assertEqual(nt1._nested_tensor_strides(), nt2._nested_tensor_strides())
            self.assertEqual(nt1._nested_tensor_storage_offsets(),
                             nt2._nested_tensor_storage_offsets())

        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))
        for a in [nt_contiguous, nt_noncontiguous]:
            buffer = io.BytesIO()
            serialized = torch.save(a, buffer)
            buffer.seek(0)
            b = torch.load(buffer, weights_only=weights_only)
            # should be both conceptually equal and metadata equivalent
            self.assertEqual(a, b)
            compare_metadata(a, b)
            # should be conceptually equal but not necessarily metadata equivalent
            self.assertEqual(b, nt_contiguous)
            self.assertEqual(b, nt_noncontiguous)

    @unittest.skipIf(PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property")
    @onlyCUDA
    def test_pin_memory(self, device):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))
        for nt in [nt_contiguous, nt_noncontiguous]:
            self.assertFalse(nt.is_pinned())
            pinned = nt.pin_memory(device)
            self.assertTrue(pinned.is_pinned())
            self.assertEqual(nt, pinned)
            self.assertNotEqual(nt.data_ptr(), pinned.data_ptr())
            # test that pin_memory on already pinned tensor has no effect
            self.assertIs(pinned, pinned.pin_memory())
            self.assertEqual(pinned.data_ptr(), pinned.pin_memory().data_ptr())

    @torch.compiler.disable
    def _validate_nt(self, nt, device, dtype, layout, requires_grad, dim, batch_size, base=None):
        # Validate a bunch of properties after NT construction.
        device = torch.device(device)
        self.assertEqual(nt.dim(), dim)
        self.assertEqual(nt.device, device)
        self.assertEqual(nt.dtype, dtype)
        self.assertEqual(nt.layout, layout)
        self.assertEqual(nt.requires_grad, requires_grad)

        if layout == torch.jagged:
            self.assertEqual(nt._values.device, device)
            self.assertEqual(nt._offsets.device, device)
            self.assertEqual(nt.shape[0], batch_size)
            # self.assertTrue(isinstance(nt.shape[1], torch.SymInt))

        # if base is not None:
        #     self.assertTrue(nt._is_view() and nt._base is base)

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_jagged_layout_construction_nested_tensor(
            self, device, dtype, requires_grad, components_require_grad):
        for tensor_list in self._get_example_tensor_lists(
                include_list_of_lists=True, include_requires_grad=components_require_grad):
            nt = torch.nested.nested_tensor2(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad)

            expected_dim = torch.as_tensor(tensor_list[0]).dim() + 1
            expected_batch_size = len(tensor_list)
            self._validate_nt(
                nt, device, dtype, torch.jagged, requires_grad, expected_dim, expected_batch_size)

            # Make sure grads -don't- flow back into original tensors for nested_tensor()
            if requires_grad:
                (nt * 2).backward(torch.ones_like(nt))
            for t in tensor_list:
                t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                self.assertTrue(t.grad is None)

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("components_require_grad", [False, True])
    def test_jagged_layout_construction_as_nested_tensor(
            self, device, dtype, components_require_grad):
        # NB: as_nested_tensor2(tensor_list) doesn't support lists of lists for tensor_list
        for tensor_list in self._get_example_tensor_lists(
                include_list_of_lists=False, include_requires_grad=components_require_grad):
            nt = torch.nested.as_nested_tensor2(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged)

            # nt.requires_grad=True should be set if at least one component requires grad
            expected_dim = tensor_list[0].dim() + 1
            expected_batch_size = len(tensor_list)
            self._validate_nt(
                nt,
                device,
                dtype,
                torch.jagged,
                components_require_grad,
                expected_dim,
                expected_batch_size)

            # Make sure grads flow back into original tensors for as_nested_tensor2()
            if components_require_grad:
                (nt * 2).backward(torch.ones_like(nt))
                for t in tensor_list:
                    if t.requires_grad:
                        self.assertEqual(t.grad, torch.ones_like(t) * 2)
                    else:
                        self.assertTrue(t.grad is None)

    @xfailIfTorchDynamo
    @unittest.skipIf(PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property")
    @onlyCUDA
    def test_jagged_layout_construction_with_pinned_memory(self, device):
        for tensor_list in self._get_example_tensor_lists():
            nt = torch.nested.nested_tensor2(
                tensor_list,
                layout=torch.jagged,
                device="cpu",
                pin_memory=True)

            expected_dim = torch.as_tensor(tensor_list[0]).dim() + 1
            expected_batch_size = len(tensor_list)
            self._validate_nt(
                nt,
                device="cpu",
                dtype=torch.float32,
                layout=torch.jagged,
                requires_grad=False,
                dim=expected_dim,
                batch_size=expected_batch_size)
            self.assertTrue(nt.is_pinned())

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("requires_grad", [False, True])
    @parametrize("values_is_view", [False, True])
    def test_jagged_view_from_values_offsets(self, device, dtype, requires_grad, values_is_view):
        if values_is_view:
            # make values a view of base
            base = torch.randn(
                2, 3, 4, 5, 6, device=device, dtype=dtype, requires_grad=requires_grad)
            values = base.flatten(0, -2)
        else:
            values = torch.randn(10, 5, device=device, dtype=dtype, requires_grad=requires_grad)
        offsets = torch.tensor([0, 2, 4, 6, 10], device=device, dtype=torch.int64)

        nt = nested_view_from_values_offsets(values, offsets)

        expected_dim = values.dim() + 1
        expected_batch_size = offsets.shape[0] - 1
        expected_base = base if values_is_view else values
        self._validate_nt(
            nt, device, dtype, torch.jagged, requires_grad, expected_dim, expected_batch_size,
            # ensure NT is a proper view
            base=expected_base
        )

        if requires_grad:
            # Make sure grads flow back
            (nt * 2).backward(torch.ones_like(nt))

            @torch.compiler.disable
            def _check_grad(t):
                self.assertTrue(t.grad is not None)
                self.assertEqual(t.grad, torch.ones_like(t) * 2)

            _check_grad(base if values_is_view else values)

    @dtypes(torch.float)
    def test_nested_tensor_from_jagged(self, device, dtype):
        # construct from (values, offsets)
        values = torch.randn(10, 5, device=device, dtype=dtype)
        offsets = torch.tensor([0, 2, 4, 6, 10], device=device, dtype=torch.int64)
        nt = torch.nested.nested_tensor_from_jagged(values, offsets=offsets)
        self.assertTrue(isinstance(nt, NestedTensor))
        self.assertTrue(nt._is_view() and nt._base is values)
        self.assertEqual(nt.dim(), 3)
        self.assertEqual(nt.size(0), offsets.size(0) - 1)
        self.assertEqual(nt.size(-1), values.size(-1))
        self.assertIsNone(nt._lengths)
        self.assertTrue(nt.is_contiguous())

        # construct from (values, offsets, lengths)
        lengths = torch.tensor([2, 1, 1, 2], device=device)
        nt = torch.nested.nested_tensor_from_jagged(values, offsets=offsets, lengths=lengths)
        self.assertTrue(isinstance(nt, NestedTensor))
        self.assertTrue(nt._is_view() and nt._base is values)
        self.assertEqual(nt.dim(), 3)
        self.assertEqual(nt.size(0), offsets.size(0) - 1)
        self.assertEqual(nt.size(-1), values.size(-1))
        self.assertEqual(nt._lengths, lengths)
        # when both offsets / lengths are specified, expect non-contiguous
        self.assertFalse(nt.is_contiguous())

        # construct from (values, lengths)
        values = torch.randn(14, 5, device=device, dtype=dtype)
        lengths = torch.tensor([2, 3, 4, 5], device=device)
        nt = torch.nested.nested_tensor_from_jagged(values, lengths=lengths)
        self.assertTrue(isinstance(nt, NestedTensor))
        self.assertTrue(nt._is_view() and nt._base is values)
        self.assertEqual(nt.dim(), 3)
        self.assertEqual(nt.size(0), lengths.size(0))
        self.assertEqual(nt.size(-1), values.size(-1))
        # for now, if only lengths is specified, convert to offsets to integrate best with the
        # existing kernels
        expected_offsets = torch.tensor([0, 2, 5, 9, 14], device=device)
        expected_nt = torch.nested.nested_tensor_from_jagged(values, offsets=expected_offsets)
        for n1, n2 in zip(nt.unbind(), expected_nt.unbind()):
            self.assertEqual(n1, n2)

        # error case: no offsets or lengths
        with self.assertRaisesRegex(RuntimeError, "At least one of offsets or lengths is required"):
            torch.nested.nested_tensor_from_jagged(values, offsets=None, lengths=None)

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("dim", range(5))
    @parametrize("layout", [torch.jagged],
                 name_fn=lambda l: f"layout_{str(l).split('.')[1]}")
    # @parametrize("layout", [torch.strided, torch.jagged],
    #              name_fn=lambda l: f"layout_{str(l).split('.')[1]}")
    @parametrize("requires_grad", [False, True])
    @parametrize("contiguous", [False, True])
    def test_as_nested_tensor_from_tensor(
            self, device, dtype, dim, layout, requires_grad, contiguous):
        if dim == 0:
            t = torch.tensor(3., requires_grad=requires_grad)
        else:
            t = torch.randn(*(3 for _ in range(dim)), requires_grad=requires_grad)
        assert t.dim() == dim

        if dim < 2:
            # 0-1 dim tensors can't be converted to NTs
            with self.assertRaisesRegex(RuntimeError, "Expected tensor argument to have dim"):
                nt = torch.nested.as_nested_tensor2(t, device=device, dtype=dtype, layout=layout)
            return

        orig_t = t
        if not contiguous:
            t = t.transpose(0, 1)

        nt = torch.nested.as_nested_tensor2(t, device=device, dtype=dtype, layout=layout)
        expected_dim = t.dim()
        expected_batch_size = t.size(0)
        self._validate_nt(
            nt, device, dtype, layout, requires_grad, expected_dim, expected_batch_size)

        # if torch.device(device) == t.device and dtype == t.dtype and contiguous:
        #     # should be the non-copying (view) case
        #     self.assertTrue(nt._is_view() and nt._base is t)

        # should be equivalent to construction from unbound tensor list
        nt_from_unbind = torch.nested.as_nested_tensor2(
            list(t.unbind(0)), device=device, dtype=dtype, layout=layout)
        self.assertEqual(nt, nt_from_unbind)

        # ensure call on a NT with the same properties returns the NT directly
        # TODO(rzou): It's OK, semantics changed
        # nt2 = torch.nested.as_nested_tensor2(nt, device=device, dtype=dtype, layout=layout)
        # self.assertTrue(nt is nt2)

        # we don't support conversion between layouts this way atm
        other_layout = torch.strided if layout == torch.jagged else torch.jagged
        with self.assertRaisesRegex(
                RuntimeError, "Converting between nested tensor layouts is not supported"):
            torch.nested.as_nested_tensor2(nt, device=device, dtype=dtype, layout=other_layout)

        if requires_grad:
            # make sure gradients flow back into inputs
            (nt * 2).backward(torch.ones_like(nt))
            self.assertEqual(orig_t.grad, torch.ones_like(orig_t) * 2)

    @dtypes(torch.double, torch.half)
    @onlyCUDA
    def test_device_dtype_transfer_updates_offsets(self, device, dtype):
        for tensor_list in self._get_example_tensor_lists():
            orig_device = torch.device("cpu")
            orig_dtype = torch.float32
            nt = torch.nested.nested_tensor2(
                tensor_list,
                layout=torch.jagged,
                device=orig_device,
                dtype=orig_dtype)

            self.assertEqual(torch.int64, nt.offsets().dtype)
            nt = nt.to(device=device).to(dtype=dtype)

            # offsets should still be int64 on the new device
            self.assertEqual(nt.values().device, nt.offsets().device)
            self.assertEqual(torch.int64, nt.offsets().dtype)

    def test_unbind(self, device):
        for tensor_list in self._get_example_tensor_lists():
            nt = torch.nested.nested_tensor2(
                tensor_list,
                layout=torch.jagged,
                device=device)
            out = nt.unbind()
            self.assertEqual(len(out), len(tensor_list))
            for i, t in enumerate(out):
                self.assertEqual(t, tensor_list[i])

    def test_layer_norm_2(self, device):
        test_tensor_list = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 3), device=device, requires_grad=True
        )
        bias = torch.randn(3, requires_grad=False, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, bias):
            nt = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)
            out = torch.nn.functional.layer_norm(nt, (nt.shape[-1],), bias=bias)
            return out.values()

        gradcheck(
            grad_test_func, inputs=(*test_tensor_list, bias), check_batched_grad=False
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"layer_norm\(\): normalizing over ragged dim not supported for nested tensors",
        ):
            nt = torch.nested.as_nested_tensor2(test_tensor_list, layout=torch.jagged)
            _ = torch.nn.functional.layer_norm(nt, (nt.shape[-2], nt.shape[-1]))

    def test_narrow(self, device):
        starts = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.int64)
        lengths = torch.tensor([3, 2, 2, 1, 5], device=device, dtype=torch.int64)
        buffer = (
            torch.arange(0, 10, device=device, dtype=torch.int64)
            .unsqueeze(0).expand(5, -1).clone().detach()
        )
        nt = torch.nested.narrow(
            buffer,
            1,
            starts,
            lengths,
            layout=torch.jagged
        )

        self.assertTrue(nt._is_view() and nt._base is buffer)

        # TODO: Use this approach when unbind is functional
        # unbinded_nt = nt.unbind()
        # for i in range(starts.shape[0]):
        #     self.assertEqual(torch.arange(starts[i], starts[i] + lengths[i], device=device, dtype=torch.int64), unbinded_nt[i])
        for i in range(starts.shape[0]):
            self.assertEqual(
                torch.arange(starts[i], starts[i] + lengths[i], device=device, dtype=torch.int64),
                nt.values()[nt.offsets()[i]:(nt.offsets()[i] + nt.lengths()[i])]
            )

    def test_is_contiguous(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)
        nt_contiguous = torch.nested.as_nested_tensor2([a, b, c], layout=torch.jagged)

        starts_nc = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.int64)
        lengths_nc = torch.tensor([3, 2, 2, 1, 5], device=device, dtype=torch.int64)
        narrow_base = torch.arange(0, 10, device=device, dtype=torch.int64).unsqueeze(0).expand(5, -1).clone()
        nt_noncontiguous = torch.nested.narrow(
            narrow_base,
            1,
            starts_nc,
            lengths_nc,
            layout=torch.jagged
        )

        starts_c = torch.tensor([1, 0, 0, 0, 0], device=device, dtype=torch.int64)
        lengths_c = torch.tensor([9, 10, 10, 10, 8], device=device, dtype=torch.int64)
        nt_contiguous_narrow = torch.nested.narrow(
            narrow_base,
            1,
            starts_c,
            lengths_c,
            layout=torch.jagged
        )

        # Test contiguous case
        assert nt_contiguous.is_contiguous()

        # Test narrow case
        assert not nt_noncontiguous.is_contiguous()
        assert nt_contiguous_narrow.is_contiguous()

        # Test querying by memory_format
        self.assertTrue(nt_contiguous.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(not nt_noncontiguous.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(nt_contiguous_narrow.is_contiguous(memory_format=torch.contiguous_format))

    def test_layout_under_torch_dispatch_mode(self):
        from torch.testing._internal.logging_tensor import capture_logs_with_logging_tensor_mode

        nt = random_nt_from_dims([2, None, 3], torch.device('cpu'), torch.float32, layout=torch.jagged)

        with capture_logs_with_logging_tensor_mode():
            self.assertEqual(nt.layout, torch.jagged)

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @parametrize("func", [torch.empty_like, torch.randn_like],
                 name_fn=lambda f: f.__name__)
    def test_like_shape(self, func):
        nt = random_nt_from_dims([2, None, 3], torch.device('cpu'), torch.float32, layout=torch.jagged)
        nt_like = func(nt)

        for nt_ub in nt_like.unbind():
            t_like = func(nt_ub)
            self.assertEqual(nt_ub.shape, t_like.shape)

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @parametrize("func", [torch.ones_like, torch.zeros_like],
                 name_fn=lambda f: f.__name__)
    def test_like_value(self, func):
        nt = random_nt_from_dims([2, None, 3], torch.device('cpu'), torch.float32, layout=torch.jagged)
        nt_like = func(nt)

        for nt_ub in nt_like.unbind():
            t_like = func(nt_ub)
            self.assertEqual(nt_ub, t_like)

    def test_noncontiguous_pointwise(self, device):
        a = torch.randn(2, 3, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, 4, requires_grad=True, dtype=torch.float64, device=device)
        nt = torch.nested.nested_tensor2([a, b, c], layout=torch.jagged)
        # transpose ragged dim
        transposed = nt.transpose(1, 2)
        self.assertFalse(transposed.is_contiguous())
        clone = transposed.clone()

        def check_nt_equality(x, y):
            self.assertEqual(x.values(), y.values())
            self.assertEqual(x.offsets(), y.offsets())
            self.assertEqual(x._ragged_idx, y._ragged_idx)
            self.assertEqual(x.shape, y.shape)

        self.assertFalse(clone.is_contiguous())
        check_nt_equality(clone, transposed)

        clone_contig = transposed.clone(memory_format=torch.contiguous_format)
        self.assertTrue(clone_contig.is_contiguous())
        check_nt_equality(clone_contig, transposed)

        detached = transposed.detach()
        self.assertFalse(clone.is_contiguous())
        check_nt_equality(detached, transposed)

    def test_to_copy(self, device):
        nt = torch.nested.nested_tensor2(
            [torch.randn(i + 2, 3, 4, requires_grad=True, dtype=torch.float64, device=device)
             for i in range(3)], layout=torch.jagged
        )

        nt_copy_dtype = torch.ops.aten._to_copy(nt, dtype=torch.float16)
        self.assertEqual(torch.float16, nt_copy_dtype.dtype)

        nt_t = nt.transpose(1, 2)
        nt_t_copy_dtype = torch.ops.aten._to_copy(nt_t, dtype=torch.float16)
        self.assertEqual(torch.float16, nt_t_copy_dtype.dtype)

    @skipIfTorchDynamo("Dynamo doesn't know how to trace prof.events()")
    def test_profiler_sequence_nr(self):
        with torch.profiler.profile() as prof:
            values = torch.randn(4, 6, requires_grad=True)
            offsets = torch.tensor([0, 2, 4])
            values = values * 2
            l = torch.nn.Linear(6, 8)
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)

            nt = l(nt)
            val = nt.values()

            loss = val.sum()
            loss.backward()

        fwd_seq_nrs = []
        for evt in prof.events():
            if "linear" in evt.name.lower() and "backward" not in evt.name.lower() and evt.sequence_nr != -1:
                fwd_seq_nrs.append(evt.sequence_nr)

        bwd_seq_nrs = []
        for evt in prof.events():
            if (
                "linear" in evt.name.lower() and
                "backward" in evt.name.lower() and
                "evaluate_function" not in evt.name.lower() and
                evt.sequence_nr != -1
            ):
                bwd_seq_nrs.append(evt.sequence_nr)

        # There should only be one such event with a sequence number:
        # the PythonTLSSnapshot event - but, note that it's not terrible if
        # we end up with multiple events with the same sequence number - so we
        # could relax this check if it becomes inconvenient to maintain this
        # property.
        self.assertEqual(len(fwd_seq_nrs), 1)
        self.assertEqual(len(bwd_seq_nrs), 1)
        self.assertEqual(fwd_seq_nrs[0], bwd_seq_nrs[0])

    def test_is_same_size(self, device):
        def get_3_tensors():
            return [torch.randn(i + 2, 3, 4, requires_grad=True, dtype=torch.float64, device=device) for i in range(3)]

        nt1, offsets1 = jagged_from_list2(get_3_tensors(), None)
        nt2, offsets1 = jagged_from_list2(get_3_tensors(), offsets1)

        nt3, offsets2 = jagged_from_list2(get_3_tensors(), None)
        nt4, offsets2 = jagged_from_list2(get_3_tensors(), offsets2)

        def check_size(nt1, nt2, nt3, nt4):
            self.assertTrue(torch.ops.aten.is_same_size(nt1, nt2))
            self.assertTrue(torch.ops.aten.is_same_size(nt3, nt4))
            self.assertFalse(torch.ops.aten.is_same_size(nt1, nt3))

        check_size(nt1, nt2, nt3, nt4)

        nt1_t, nt2_t, nt3_t, nt4_t = (x.transpose(1, 2) for x in (nt1, nt2, nt3, nt4))
        check_size(nt1_t, nt2_t, nt3_t, nt4_t)

    # Doesn't work until we have real views
    @xfailIfTorchDynamo
    # Note 1: Math fallback doesn't work with bfloat16 on CUDA
    # Note 2: ROCm doesn't support flash attention or mem_efficient attention for NT
    @unittest.skipIf(
        TEST_WITH_ROCM,
        "ROCm doesn't support flash attention or mem_efficient attention for NT",
    )
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32] if
                 SM80OrLater else [torch.float16, torch.float32])
    def test_sdpa(self, device, dtype):
        batch_size = 1
        emb_dims = 128
        n_heads = 8
        head_dims = emb_dims // n_heads

        sen1 = torch.randn(11, emb_dims, dtype=dtype, device=device)
        sen2 = torch.randn(13, emb_dims, dtype=dtype, device=device)

        query = torch.nn.Linear(emb_dims, emb_dims, bias=False, device=device, dtype=dtype)
        key = torch.nn.Linear(emb_dims, emb_dims, bias=False, device=device, dtype=dtype)
        value = torch.nn.Linear(emb_dims, emb_dims, bias=False, device=device, dtype=dtype)

        # Simplest case: 1 sentence, no batching
        x_d1 = sen1.unsqueeze(0)
        x_nt = torch.nested.as_nested_tensor2([sen1], layout=torch.jagged)

        # See note below for why we detach here.
        q_d1 = query(x_d1).view(batch_size, -1, n_heads, head_dims).detach().requires_grad_(True)
        q_d1_t = q_d1.transpose(1, 2)
        k_d1 = key(x_d1).view(batch_size, -1, n_heads, head_dims).detach().requires_grad_(True)
        k_d1_t = k_d1.transpose(1, 2)
        v_d1 = value(x_d1).view(batch_size, -1, n_heads, head_dims).detach().requires_grad_(True)
        v_d1_t = v_d1.transpose(1, 2)

        q_nt = query(x_nt).view(*x_nt.size()[0:2], n_heads, head_dims).detach().requires_grad_(True)
        q_nt_t = q_nt.transpose(1, 2)
        k_nt = key(x_nt).view(*x_nt.size()[0:2], n_heads, head_dims).detach().requires_grad_(True)
        k_nt_t = k_nt.transpose(1, 2)
        v_nt = value(x_nt).view(*x_nt.size()[0:2], n_heads, head_dims).detach().requires_grad_(True)
        v_nt_t = v_nt.transpose(1, 2)

        # High Precision Math Reference
        q_d1_f32 = q_d1.to(torch.float32)
        k_d1_f32 = k_d1.to(torch.float32)
        v_d1_f32 = v_d1.to(torch.float32)
        q_d1_f32_t = q_d1_f32.transpose(1, 2)
        k_d1_f32_t = k_d1_f32.transpose(1, 2)
        v_d1_f32_t = v_d1_f32.transpose(1, 2)
        out_ref = torch.ops.aten._scaled_dot_product_attention_math(q_d1_f32_t, k_d1_f32_t, v_d1_f32_t)[0]
        grads_ref = torch.autograd.grad(out_ref.sum(), (q_d1_f32, k_d1_f32, v_d1_f32))

        # Low Precision Math Reference
        out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(q_d1_t, k_d1_t, v_d1_t)[0]
        grads_lp_ref = torch.autograd.grad(out_lp_ref.sum(), (q_d1, k_d1, v_d1))

        # Compute tolerances
        output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)
        grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(grads_ref[0], grads_lp_ref[0])
        grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(grads_ref[1], grads_lp_ref[1])
        grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(grads_ref[2], grads_lp_ref[2])
        grad_atols = [grad_q_ref_atol, grad_k_ref_atol, grad_v_ref_atol]
        grad_rtols = [grad_q_ref_rtol, grad_k_ref_rtol, grad_v_ref_rtol]

        attn_d1 = torch.nn.functional.scaled_dot_product_attention(q_d1_t, k_d1_t, v_d1_t).transpose(1, 2)
        attn_nt = torch.nn.functional.scaled_dot_product_attention(q_nt_t, k_nt_t, v_nt_t).transpose(1, 2)

        self.assertEqual(attn_d1, attn_nt.unbind()[0].unsqueeze(0), atol=output_ref_atol, rtol=output_ref_rtol)

        # Simple case: 2 sentences, no extra params
        x_d2 = sen2.unsqueeze(0)
        x_nt = torch.nested.as_nested_tensor2([sen1, sen2], layout=torch.jagged)

        # NB: we make sure the leaf tensor we compute gradients for is the view-ed tensor before
        # it is transposed. This is because today we cannot backward through view or unbind a
        # transposed tensor.
        q_d2 = query(x_d2).view(batch_size, -1, n_heads, head_dims).detach().requires_grad_(True)
        q_d2_t = q_d2.transpose(1, 2)
        k_d2 = key(x_d2).view(batch_size, -1, n_heads, head_dims).detach().requires_grad_(True)
        k_d2_t = k_d2.transpose(1, 2)
        v_d2 = value(x_d2).view(batch_size, -1, n_heads, head_dims).detach().requires_grad_(True)
        v_d2_t = v_d2.transpose(1, 2)

        q_nt = query(x_nt).view(*x_nt.size()[0:2], n_heads, head_dims).detach().requires_grad_(True)
        q_nt_t = q_nt.transpose(1, 2)
        k_nt = key(x_nt).view(*x_nt.size()[0:2], n_heads, head_dims).detach().requires_grad_(True)
        k_nt_t = k_nt.transpose(1, 2)
        v_nt = value(x_nt).view(*x_nt.size()[0:2], n_heads, head_dims).detach().requires_grad_(True)
        v_nt_t = v_nt.transpose(1, 2)

        attn_d2 = torch.nn.functional.scaled_dot_product_attention(q_d2_t, k_d2_t, v_d2_t).transpose(1, 2)
        d1_grads = torch.autograd.grad(attn_d1.sum(), (q_d1, k_d1, v_d1))
        d2_grads = torch.autograd.grad(attn_d2.sum(), (q_d2, k_d2, v_d2))

        def check_forward_backward():
            attn_nt = torch.nn.functional.scaled_dot_product_attention(q_nt_t, k_nt_t, v_nt_t).transpose(1, 2)

            attn_nts = attn_nt.unbind()
            self.assertEqual(attn_d1, attn_nts[0].unsqueeze(0), atol=output_ref_atol, rtol=output_ref_rtol)
            self.assertEqual(attn_d2, attn_nts[1].unsqueeze(0), atol=output_ref_atol, rtol=output_ref_rtol)

            nt_grads = torch.autograd.grad(attn_nt.values().sum(), (q_nt, k_nt, v_nt))
            for nt_grad, d1_grad, d2_grad, grad_atol, grad_rtol in zip(nt_grads, d1_grads, d2_grads, grad_atols, grad_rtols):
                unbound_nt_grads = nt_grad.unbind()
                self.assertEqual(d1_grad, unbound_nt_grads[0].unsqueeze(0), atol=grad_atol, rtol=grad_rtol)
                self.assertEqual(d2_grad, unbound_nt_grads[1].unsqueeze(0), atol=grad_atol, rtol=grad_rtol)

        # Default
        check_forward_backward()

        # Test dispatcher works by calling only mem-effn and math (as they are safe for all devices)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
            check_forward_backward()

        # Test math fallback
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            # Math fallback doesn't work with bfloat16 on CUDA because
            # "group_gemm_dispatch" not implemented for 'BFloat16'
            if not (str(device).startswith("cuda") and dtype == torch.bfloat16):
                check_forward_backward()

    @skipIfTorchDynamo("SDPA test compiles internally")
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    # Guarding with sqrt() doesn't work on ROCm?
    @skipCUDAIfRocm
    @onlyCUDA
    @dtypes(*([torch.float16, torch.bfloat16, torch.float32] if SM80OrLater
            else [torch.float16, torch.float32]))
    def test_sdpa_compile(self, device, dtype):
        batch_size = 1
        emb_dims = 1024
        n_heads = 8
        head_dims = emb_dims // n_heads

        sen1 = torch.randn(11, emb_dims, dtype=dtype, device=device)
        sen2 = torch.randn(13, emb_dims, dtype=dtype, device=device)

        query = torch.nn.Linear(emb_dims, emb_dims, bias=False, device=device, dtype=dtype)
        key = torch.nn.Linear(emb_dims, emb_dims, bias=False, device=device, dtype=dtype)
        value = torch.nn.Linear(emb_dims, emb_dims, bias=False, device=device, dtype=dtype)

        # Simplest case: 1 sentence, no batching
        x_d1 = sen1.unsqueeze(0)
        x_d2 = sen2.unsqueeze(0)
        x_nt = torch.nested.as_nested_tensor2([sen1, sen2], layout=torch.jagged)

        q_d1 = query(x_d1).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)
        k_d1 = key(x_d1).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)
        v_d1 = value(x_d1).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)
        q_d2 = query(x_d2).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)
        k_d2 = key(x_d2).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)
        v_d2 = value(x_d2).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)

        q_nt = query(x_nt).view(*x_nt.size()[0:2], n_heads, head_dims).detach().transpose(1, 2)
        k_nt = key(x_nt).view(*x_nt.size()[0:2], n_heads, head_dims).detach().transpose(1, 2)
        v_nt = value(x_nt).view(*x_nt.size()[0:2], n_heads, head_dims).detach().transpose(1, 2)

        # High Precision Math Reference
        q_d1_f32 = q_d1.to(torch.float32)
        k_d1_f32 = k_d1.to(torch.float32)
        v_d1_f32 = v_d1.to(torch.float32)
        out_ref = torch.ops.aten._scaled_dot_product_attention_math(q_d1_f32, k_d1_f32, v_d1_f32)[0]
        # Low Precision Math Reference
        out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(q_d1, k_d1, v_d1)[0]
        output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)

        attn_d1 = torch.nn.functional.scaled_dot_product_attention(q_d1, k_d1, v_d1).transpose(1, 2)
        attn_d2 = torch.nn.functional.scaled_dot_product_attention(q_d2, k_d2, v_d2).transpose(1, 2)

        compiled_sdpa = torch.compile(torch.nn.functional.scaled_dot_product_attention)
        attn_nt = compiled_sdpa(q_nt, k_nt, v_nt).transpose(1, 2)

        attn_nts = attn_nt.unbind()
        self.assertEqual(attn_d1, attn_nts[0].unsqueeze(0), atol=output_ref_atol, rtol=output_ref_rtol)
        self.assertEqual(attn_d2, attn_nts[1].unsqueeze(0), atol=output_ref_atol, rtol=output_ref_rtol)

    @dtypes(torch.float32, torch.double, torch.half)
    def test_sdpa_with_constant_sequence_length(self, device, dtype):
        # shape (B, P*, S, D)
        # B: batch size
        # P*: ragged number of prompts
        # S: (constant) sequence length
        # D: embedding size
        query = random_nt_from_dims(
            [4, None, 8, 10], device=device, dtype=dtype, layout=torch.jagged)
        key = random_nt_from_similar(query)
        value = random_nt_from_similar(query)
        output = F.scaled_dot_product_attention(query, key, value)
        # self.assertTrue(isinstance(output, NestedTensor))

        # should be equivalent to just running the buffers through
        output_dense = F.scaled_dot_product_attention(query._values, key._values, value._values)
        self.assertEqual(output._values, output_dense)

    # Doesn't work until we have real views
    @xfailIfTorchDynamo
    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Platform doesn't support flash or mem-efficient attention"
    )
    @dtypes(*([torch.float16, torch.bfloat16, torch.float32] if SM80OrLater
            else [torch.float16, torch.float32]))
    def test_sdpa_with_packed_in_proj(self, device, dtype):
        # shape (B, *, D)
        input_packed = random_nt_from_dims(
            [5, None, 10], device=device, dtype=dtype, layout=torch.jagged)

        # Do input projection.
        num_heads = 2
        # should be multiple of 4 for efficient kernels (e.g. flash / mem-efficient)
        head_dim = 8
        qkv_linear = torch.nn.Linear(10, num_heads * head_dim * 3).to(device=device, dtype=dtype)

        def in_proj(input_packed, qkv_linear=qkv_linear):
            qkv_post_proj = qkv_linear(input_packed)
            # these are non-contiguous to trigger _is_safe_to_get_storage_as_tensor()
            q, k, v = qkv_post_proj.chunk(3, dim=-1)
            q = q.unflatten(-1, [num_heads, head_dim]).transpose(-2, -3)
            k = k.unflatten(-1, [num_heads, head_dim]).transpose(-2, -3)
            v = v.unflatten(-1, [num_heads, head_dim]).transpose(-2, -3)
            return q, k, v

        q, k, v = in_proj(input_packed)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=None)

        # compare to individually running unbound components through
        for in_component, out_component in zip(
            input_packed.unbind(),
            output.transpose(-2, -3).unbind()
        ):
            q, k, v = in_proj(in_component)
            out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)

            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                q, k, v)[0].transpose(-2, -3)
            output_ref_atol, output_ref_rtol = get_tolerances(out, out_lp_ref, fudge_factor=2)

            self.assertEqual(out, out_component, atol=output_ref_atol, rtol=output_ref_rtol)

    @skipIfTorchDynamo("SDPA test compiles internally")
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    # mha_varlen_fwd not supported on ROCm
    @skipCUDAIfRocm
    @onlyCUDA
    @dtypes(*([torch.float16, torch.bfloat16, torch.float32] if SM80OrLater
            else [torch.float16, torch.float32]))
    def test_sdpa_backwards(self, device, dtype):
        values = torch.randn(9, 3, 256, requires_grad=True, device=device, dtype=dtype)
        offsets = torch.tensor([0, 1, 3, 5, 9], device=device, dtype=torch.int64)

        @torch.compile
        def f(values, offsets):
            nt = convert_jagged_to_nested_tensor(values, offsets, max_length=4)
            nt = nt.transpose(-2, -3)
            # purposefully graph break to trigger view replay for subclass view input
            torch.tensor(1).item()
            output = F.scaled_dot_product_attention(nt, nt, nt).transpose(-2, -3)
            return convert_nt_to_jagged(output)

        output = f(values, offsets)
        output.sum().backward()
        self.assertEqual(values.grad, torch.ones_like(values))

    # Internally-defined NT use cases are lifted to here for maximum test realism.
    # TODO: Remove these when ViewNestedFromBuffer, etc. are deprecated.
    @skipCUDAIfRocm  # not needed
    @skipIfTorchDynamo("compiles internally")
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    def test_dummy_mha_with_nt(self, device):
        bs = 3
        d1 = 2
        d2 = 4
        d3 = 6
        n_heads = 2
        d_head = d3 // n_heads
        max_length_1 = 10
        max_length_2 = 20
        torch.manual_seed(0)

        class mha(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.linear = torch.nn.Linear(d2, d3, device=device)

            def forward(self, query, value, offsets):

                value = self.linear(value)
                key = convert_jagged_to_nested_tensor(value, offsets, max_length_1)
                value = convert_jagged_to_nested_tensor(value, offsets, max_length_2)
                query = convert_dense_to_nested_tensor(query)
                q = query.view(bs, -1, n_heads, d_head).transpose(1, 2)
                k = key.view(bs, -1, n_heads, d_head).transpose(1, 2)
                v = value.view(bs, -1, n_heads, d_head).transpose(1, 2)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
                attn_output = attn_output.transpose(1, 2)
                attn_output = convert_nt_to_jagged(attn_output)
                return attn_output, key._max_seqlen, value._max_seqlen

        query = torch.rand(bs, d1, d3, device=device)
        value = torch.rand(6, d2, requires_grad=True, device=device)
        offsets = torch.tensor([0, 2, 3, 6], device=device)

        m = mha()
        symbolic_traced: torch.fx.GraphModule = torch.fx.symbolic_trace(m)
        m = torch.compile(symbolic_traced)
        attn_output, cached_key_max_seqlen, cached_value_max_seqlen = m(
            query, value, offsets
        )
        loss = attn_output.sum()
        # Check that NT can be fx traced and torch.compile, and backward works
        loss.backward()

        # Check that value.requires_grad is not lost after tracing and compiling
        value_grad = value.grad  # save for comparison later
        self.assertIsNotNone(value_grad)
        # check that max_seqlen is cached properly
        self.assertEqual(cached_key_max_seqlen, max_length_1)
        self.assertEqual(cached_value_max_seqlen, max_length_2)

        # check if the output is numerically equivalent with the eager mode
        m_eager = mha()
        value.grad = None
        attn_output_eager, _, _ = m_eager(query, value, offsets)
        attn_output_eager.sum().backward()
        self.assertTrue(torch.allclose(attn_output_eager, attn_output))
        self.assertTrue(torch.allclose(value_grad, value.grad))


instantiate_device_type_tests(TestNestedTensorSubclass, globals())

if __name__ == '__main__':
    run_tests()
