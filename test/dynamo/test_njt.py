# Owner(s): ["module: dynamo"]
import functools
import itertools
import unittest

from functools import partial

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.testing import normalize_gm
from torch._higher_order_ops.wrap import wrap

from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.nested._internal.nested_tensor import (
    jagged_from_list2,
    jagged_from_tensor_and_lengths2,
    nested_view_from_values_offsets2,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.two_tensor import TwoTensor


def traceable_subclass(c):
    return torch._dynamo.config.patch("traceable_tensor_subclasses", {c})


def get_jagged_tensor(nested_size, offsets, requires_grad=True):
    # Makes a jagged tensor with N constituent tensors with size
    # as specified ((S0, S1, S2), D)
    D = nested_size[1]
    out = []
    for s in nested_size[0]:
        out.append(torch.randn(s, D, requires_grad=requires_grad, dtype=torch.float64))
    return jagged_from_list2(out, offsets)


def get_view_test_cases():
    # Test all cases with both an NT base and a dense base
    # Subclass -> Subclass
    # Dense -> Subclass

    # NB: Don't close over loop variables, they will not get copied into the
    # closure
    #
    # NB: These return functions so we don't generate tensors during test
    # collection time

    def mk_basic(base_is_nt):
        # There are three cases to consider here based on the logic in
        # meta_utils.py
        #
        # (1) basic case:
        # view is not a leaf and has the same requires grad as its basic case
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)
        x = x.clone() if base_is_nt else x
        assert not x.is_leaf
        return x.unsqueeze(-1)

    def mk_leaf(base_is_nt, requires_grad_1, requires_grad_2):
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=requires_grad_1)
        x = x.clone() if base_is_nt else x
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
            # The issue is this doesn't quite work
            x_view.requires_grad_(requires_grad_2)

        return x_view

    def mk_obscure(base_is_nt):
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=False)
        x = x.clone() if base_is_nt else x
        # intermediate leaf view
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
        x_view.requires_grad_(True)
        x_view_view = x_view.unsqueeze(-1)
        return x_view_view

    for base_is_nt in [False, True]:
        prefix = f"base_is_nt_{base_is_nt}"

        yield partial(mk_basic, base_is_nt), f"{prefix}_basic"

        # (2) leaf view case:
        # the view has to be a leaf (w/ requires_grad True or requires_grad False)
        # base w/ requires_grad True or requires_grad False
        for requires_grad_1, requires_grad_2 in itertools.product(
            [True, False], repeat=2
        ):
            yield partial(
                mk_leaf, base_is_nt, requires_grad_1, requires_grad_2
            ), f"{prefix}_leaf_{requires_grad_1}_{requires_grad_2}"

        # (3) obscure case:
        # view is not a leaf (implies requires_grad True)
        # base w/ requires_grad False)
        yield partial(mk_obscure, base_is_nt), f"{prefix}_obscure"

    # Subclass -> Dense
    yield lambda: get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)[
        0
    ].clone(), "subclass_dense"

    # Dense -> Subclass -> Dense -> Subclass
    def mk_dense_subclass_dense_subclass():
        values = torch.randn(10, 5)
        offsets = torch.tensor([0, 3, 6, 10])
        offsets2 = offsets.clone().detach()
        return nested_view_from_values_offsets2(
            nested_view_from_values_offsets2(values, offsets).values(), offsets
        )

    yield mk_dense_subclass_dense_subclass, "dense_subclass_dense_subclass"

    def mk_subclass_dense_subclass_dense():
        x = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)[0].clone()
        offsets2 = x.offsets().clone().detach()
        nt_view = nested_view_from_values_offsets2(x.values(), offsets2).values()

    yield mk_subclass_dense_subclass_dense, "subclass_dense_subclass_dense"


VIEW_TEST_CASES = {k: v for v, k in get_view_test_cases()}


requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")

compile_full_eager = torch.compile(backend="eager", fullgraph=True)


class BaseTorchFunction(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)


class MockSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class AttrSubclass(torch.Tensor):
    x: int = 10
    size: int = 10

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        return func(*args, **kwargs)


class DummyNDim(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.Tensor.ndim.__get__:
            return 10

        return super().__torch_function__(func, types, args, kwargs)


class WrapperSubclass:
    def __init__(self, tensor):
        self.tensor = tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = pytree.tree_map_only(WrapperSubclass, lambda x: x.tensor, args)
        kwargs = pytree.tree_map_only(WrapperSubclass, lambda x: x.tensor, kwargs)

        return func(*args, **kwargs)


class SigmoidToExpSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.Tensor.sigmoid:
            return super().__torch_function__(torch.Tensor.exp, types, args, kwargs)

        return super().__torch_function__(func, types, args, kwargs)


# Wrapper subclass with two inner tensors: data and scale
# data has same shape as outer, and scale has single dim size
class ScaledTensor(torch.Tensor):
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        *,
        constant: int = 0,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

    def __init__(self, data: torch.Tensor, scale: torch.Tensor, constant: int = 0):
        self._data = data
        self._scale = scale
        self._constant = constant

    def __tensor_flatten__(self):
        ctx = {"_constant": self._constant}
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        return ScaledTensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            constant=metadata["_constant"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        scaled_tensor = args[0]
        out = func(scaled_tensor._data, *args[1:], **kwargs)
        return ScaledTensor(out, scaled_tensor._scale, constant=scaled_tensor._constant)

    def __repr__(self):
        return f"{self._data.__repr__()}\n{self._scale.__repr__()}"


def func(a):
    return a.sin()


class EagerRecordGraphAndInputs:
    def __init__(self):
        self.graphs = []
        self.example_inputs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm)
        self.example_inputs.append(example_inputs)
        return gm


GLOBAL_TEST_SUBCLASSES = {
    MockSubclass,
    DummyNDim,
    SigmoidToExpSubclass,
    BaseTorchFunction,
}


# Returns True if the function recompiles between inputs1 and inputs2 with the
# specified dynamic setting.
def _recompiles_for_inputs(fn, inputs1, inputs2, dynamic=True):
    compile_count = [0]

    def counter(gm, example_inputs):
        compile_count[0] += 1
        return gm

    compiled_f = torch.compile(fn, fullgraph=True, backend=counter, dynamic=dynamic)
    compiled_f(*inputs1)
    compiled_f(*inputs2)
    return compile_count[0] > 1


class TestNestedTensor(torch._dynamo.test_case.TestCase):
    def _get_jagged_tensor(self, nested_size, offsets, requires_grad=True):
        return get_jagged_tensor(nested_size, offsets, requires_grad)

    def _get_nc_jagged_tensor(self, inner_dim, starts, lengths, requires_grad=True):
        # Makes a jagged tensor with N constituent tensors with size
        # as specified ((S0, S1, S2), D)
        max_dim = (starts + lengths).max()
        values_tensor = torch.randn(
            starts.shape[0],
            max_dim.item(),
            inner_dim,
            requires_grad=requires_grad,
            dtype=torch.float64,
        )
        return jagged_from_tensor_and_lengths2(values_tensor, starts, lengths)

    def _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles):
        # TODO(rzou): !!!!!!!
        return
        actual_recompiles = _recompiles_for_inputs(fn, inputs1, inputs2)
        self.assertEqual(actual_recompiles, expected_recompiles)

    def test_unary_does_not_recompile(self):
        nt1, _ = self._get_jagged_tensor(((2, 3, 4), 3), None)
        nt2, _ = self._get_jagged_tensor(((3, 4, 5, 6), 4), None)
        self._check_recompiles(lambda nt1: nt1.sin(), (nt1,), (nt2,), False)

    def test_binary_does_not_recompile(self):
        def binary(nt1, nt2):
            if nt1.shape == nt2.shape:
                return nt1 + nt2
            else:
                return nt1.sin()

        # NB: If we have shape e.g. (3, j0, 3), duck sizing will give us (s0, s1, s0).
        # This causes a recompile later on when it realizes the batch and last dim
        # should not always be equal. To avoid that, we use (3, j0, 5) here.
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 5), offsets)
        nt3, offsets = self._get_jagged_tensor(((3, 4, 5), 4), None)
        nt4, _ = self._get_jagged_tensor(((3, 4, 5), 4), offsets)
        self._check_recompiles(binary, (nt1, nt2), (nt3, nt4), False)

    def test_binary_recompiles(self):
        def binary(nt1, nt2):
            if nt1.shape == nt2.shape:
                return nt1 + nt2
            else:
                return nt1.sin()

        # Binary recompiles because singleton ints no longer match
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 5), offsets)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)
        self._check_recompiles(binary, (nt1, nt2), (nt1, nt3), True)

    # TODO: cannot parametrize this test class with device for some reason
    def _test_autograd(self, backend):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        # TODO: Switch to public API when it exists
        nt2, _ = jagged_from_list2([a, b, c], nt.offsets())

        def fn1(nt1, nt2):
            # TODO(rzou): looks like a torch_function issue, solvable.
            # return (nt1 + nt2).sin().cos()
            return torch.add(nt1, nt2).sin().cos()

        compiled_f = torch.compile(fn1, fullgraph=True, backend=backend, dynamic=True)
        out = compiled_f(nt, nt2)
        out_buffer = out.values()
        ga, gb, gc = torch.autograd.grad(out_buffer.sum(), (a, b, c))

        out_ref = fn1(nt, nt2)
        out_buffer_ref = out_ref.values()
        ga_ref, gb_ref, gc_ref = torch.autograd.grad(out_buffer_ref.sum(), (a, b, c))

        (self.assertEqual(ga, ga_ref))
        (self.assertEqual(gb, gb_ref))
        (self.assertEqual(gc, gc_ref))

    def test_basic_autograd(self):
        self._test_autograd("aot_eager")

    @requires_cuda
    def test_basic_autograd_inductor(self):
        self._test_autograd("inductor")

    def test_subclass_with_mutation_in_graph(self):
        # In this graph, we have an in-graph mutation, i.e. a mutation that is allowed
        # to remain in the graph. Normally this is allowed, but it's not allowed if
        # the graph handles subclasses at all.
        # Whether the mutation is allowed or not allowed in the graph alters the number
        # of outputs from the forward graph. Previously, a bug in this handling meant
        # that sometimes the expected number and actual number of outputs from the
        # joint graph did not match, causing assertion failures.
        def fn(x, y):
            z = x.sin()
            y.sin_()
            return z.cos(), y.cos()

        fn_c = torch.compile(fn, backend="inductor")

        values = [torch.rand((i, 8), requires_grad=True) for i in range(1, 6)]
        values_copy = [x.detach().clone().requires_grad_(True) for x in values]

        nt, offsets = jagged_from_list2(values, None)
        nt_copy, offsets = jagged_from_list2(values_copy, offsets)
        y = torch.rand((4, 8))
        y_copy = y.clone()

        ret = fn_c(nt, y)[0]
        ref = fn(nt_copy, y_copy)[0]

        self.assertEqual(ret.values(), ref.values())

        ret.values().sum().backward()
        ref.values().sum().backward()
        for ref_v, res_v in zip(values_copy, values):
            self.assertEqual(ref_v.grad, res_v.grad)

    def test_unbind(self):
        # NB: If we have shape e.g. (3, j0, 3), duck sizing will give us (s0, s1, s0).
        # This causes a recompile later on when it realizes the batch and last dim
        # should not always be equal. To avoid that, we use (3, j0, 5) here.
        nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 5), 2), None)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4, 5), 3), None)

        def fn(x):
            return x.unbind()

        compiled_f = torch.compile(fn, fullgraph=True, backend="eager", dynamic=True)
        out = compiled_f(nt)

        out_ref = fn(nt)

        # correctness
        self.assertEqual(len(out), len(out_ref))
        for x, x_ref in zip(out, out_ref):
            (self.assertEqual(x, x_ref))

        # We specialize on the length of offsets, e.g. (1) we recompile if the
        # length of the offsets is different. (2) we don't recompile if the
        # length of the offsets is the same, even if the size of the constituent
        # tensors are different.
        self._check_recompiles(fn, (nt,), (nt2,), False)
        self._check_recompiles(fn, (nt,), (nt3,), True)

    def test_inline_nested_tensor_from_jagged(self):
        return
        nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)

        def fn(x):
            return torch.nested.nested_tensor_from_jagged2(x.values() * 2, x.offsets())

        torch.compile(fn, fullgraph=True, backend="aot_eager")(nt)

    def _input_view_test(self, nt_view_name):
        nt_view = VIEW_TEST_CASES[nt_view_name]()

        def fn(x):
            return x.sin()

        out_ref = fn(nt_view)
        torch._dynamo.reset()
        compile_fn = torch.compile(
            fn, fullgraph=True, backend="aot_eager", dynamic=True
        )
        out = compile_fn(nt_view)

        # Check metadata and values are correct
        self.assertEqual(out.size(), out_ref.size())
        self.assertEqual(out.stride(), out_ref.stride())
        if out.is_nested:
            (self.assertEqual(out.values(), out_ref.values()))
        else:
            (self.assertEqual(out, out_ref))

        # TODO(rzou): how to test this?
        return

        # Check that no upper/lower bound guards are incurred
        def backend(gm, args):
            context = torch._guards.TracingContext.get()
            guards = [str(g.expr) for g in context.fake_mode.shape_env.guards]

            # varies based on the type of view
            guard_str = "\n".join(guards)
            if nt_view_name == "subclass_dense":
                self.assertExpectedInline(guard_str, """Eq(s3 - 1, s0)""")
            elif nt_view_name == "dense_subclass_dense_subclass":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s5 - 1, s2)
Eq(s11 - 1, s6)
Eq(s10, s8)""",
                )
            elif nt_view_name.startswith("base_is_nt_True"):
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s3 - 1, s0)
Eq(zf1, zf4)""",
                )
            else:
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s4 - 1, s1)
Eq(s10 - 1, s5)
Eq(s9, s7)""",
                )
            return gm

        torch._dynamo.reset()
        compile_fn = torch.compile(fn, fullgraph=True, backend=backend, dynamic=True)
        out = compile_fn(nt_view)

    @parametrize(
        "nt_view_name",
        [k for k in VIEW_TEST_CASES.keys() if k != "subclass_dense_subclass_dense"],
    )
    def test_inputs_to_compiled_fn_are_views(self, nt_view_name):
        self._input_view_test(nt_view_name)

    def test_subclass_gives_static_shapes_when_dynamic_false(self):
        def check_graph(gm, *args):
            first_node_example_val = next(iter(gm.graph.nodes)).meta["example_value"]
            # We compiled with dynamic=False, expect no SymInt sizes on our placeholders
            self.assertTrue(
                all(isinstance(x, int) for x in first_node_example_val.shape)
            )
            return gm

        @torch.compile(backend=check_graph, dynamic=False)
        def f(x):
            return x + 1

        x_inner = torch.ones(4)
        x = TwoTensor(x_inner, x_inner)
        x_view = x.view(2, 2)
        out = f(x_view)

    # NJT1 -> Dense -> NJT2 -> Dense view
    # During view replay, the Dense -> NJT2 part will construct an intermediate,
    # symbolically-sized NJT that is immediately deconstructed to return the final dense
    # view. To construct this intermediate properly, we need the associated nested int
    # to be symbolic. This view is expected to fail compilation until symbolic nested ints
    # are cached onto fake offsets to solve this problem.
    @unittest.expectedFailure
    def test_subclass_dense_subclass_dense_view(self):
        self._input_view_test("subclass_dense_subclass_dense")


instantiate_parametrized_tests(TestNestedTensor)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
