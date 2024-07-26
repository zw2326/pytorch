# mypy: allow-untyped-defs
import contextlib
from typing import Callable, List, Optional
from unittest.mock import patch

import torch
from .. import ir, lowering as L
from ..virtualized import V
from .cpp_gemm_template import (
    CppPackedGemmTemplate,
    GEMM_TEMPLATE,
    get_padded_n,
    MICROKERNEL_DEF,
)

from .cpp_micro_gemm import LayoutType
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import DTYPE_TO_CPP, GemmBlocking

GEMM_SINGLE_THREAD_MM_STUB = r"""
void single_thread_mm(
    const {{X_dtype}}* X,
    const {{W_dtype}}* W,
    {{Y_dtype}}* Y
    {%- if is_dynamic_M %},
    const int64_t {{kernel.size(GemmOut, -2, unwrapped=True)}}
    {%- endif %}
)
"""

GEMM_THREADED_MM_STUB = r"""
void threaded_mm(
    const {{X_dtype}}* X,
    const {{W_dtype}}* W,
    {{Y_dtype}}* Y
    {%- if is_dynamic_M %},
    const int64_t {{kernel.size(GemmOut, -2, unwrapped=True)}}
    {%- endif %}
)
"""

BMM_WRAPPER = r"""
extern "C"
{{kernel.def_kernel(inputs={"BX": BX, "BW": BW}, outputs={"BY": BY}, aliases=aliases)}}
{
    const int64_t B = {{kernel.size(BY, -3, unwrapped=True)}};
    {%- if num_threads > 1 %}
    constexpr int64_t num_threads = {{num_threads}};
    int64_t B_single_thread_block = (B / num_threads) * num_threads;

    #pragma omp parallel for num_threads({{num_threads}})
    {%- else %}
    int64_t B_single_thread_block = B;
    {%- endif %}
    for (int64_t b_start = 0; b_start < B_single_thread_block; ++b_start) {
        single_thread_mm(
            &{{kernel.index(BX, ["b_start", 0, 0])}},
            {%- if template.should_pack_weights %}
            &{{kernel.index(BW, ["b_start", 0, 0, 0])}},
            {%- else %}
            &{{kernel.index(BW, ["b_start", 0, 0])}},
            {%- endif %}
            &{{kernel.index(BY, ["b_start", 0, 0])}}
            {%- if is_dynamic_M %},
            {{kernel.size(GemmOut, -2)}}
            {%- endif %}
        );
    }
    for (int64_t b_start = B_single_thread_block; b_start < B; ++b_start) {
        threaded_mm(
            &{{kernel.index(BX, ["b_start", 0, 0])}},
            {%- if template.should_pack_weights %}
            &{{kernel.index(BW, ["b_start", 0, 0, 0])}},
            {%- else %}
            &{{kernel.index(BW, ["b_start", 0, 0])}},
            {%- endif %}
            &{{kernel.index(BY, ["b_start", 0, 0])}}
            {%- if is_dynamic_M %},
            {{kernel.size(GemmOut, -2)}}
            {%- endif %}
        );
    }
}
"""


class CppBmmTemplate(CppPackedGemmTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
        has_bias=False,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
        name="bmm",
    ):
        super().__init__(
            input_nodes,
            layout,
            num_threads,
            register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
            name=name,
        )
        # Value may be changed after micro_gemm is instantiated if using VNNI layout
        self.should_pack_weights = False

    @classmethod
    def prep_weight(cls, inputs, layout_or_out, micro_gemm):
        if isinstance(inputs[1], ir.IRNode):
            n = inputs[1].get_size()[-1]
        else:
            n = inputs[1].shape[-1]
        _, block_n, _ = micro_gemm.register_blocking
        padded_n = get_padded_n(n, block_n)
        if n != padded_n and micro_gemm.get_b_layout() == LayoutType.NORMAL:
            cls._pad_weight(inputs, padding=padded_n - n)
        elif micro_gemm.get_b_layout() != LayoutType.NORMAL:
            inputs, layout_or_out = cls._pack_weight(inputs, layout_or_out, micro_gemm)
        return inputs, layout_or_out

    @staticmethod
    def _pad_weight(inputs, padding):
        W = inputs[1]
        if isinstance(W, ir.IRNode):
            padded_w = L.constant_pad_nd(W, (0, padding))
            padded_w = ir.ExternKernel.realize_input(padded_w)
            padded_w = ir.ExternKernel.require_contiguous(padded_w)
            if isinstance(padded_w, ir.ReinterpretView):
                # normalize stride to be "contiguous_strides" per size
                # this avoids the problems in L.view during template codegen
                assert isinstance(padded_w.layout, ir.FixedLayout)
                padded_w.layout = ir.FixedLayout(
                    padded_w.layout.device,
                    padded_w.layout.dtype,
                    padded_w.layout.size,
                    ir.FlexibleLayout.contiguous_strides(padded_w.layout.size),
                    padded_w.layout.offset,
                )
        else:
            padded_w = torch.nn.functional.pad(W, (0, padding))
        inputs[1] = padded_w

    @staticmethod
    def _pack_weight(inputs, layout_or_out, micro_gemm):
        if isinstance(inputs[0], ir.IRNode):
            b, k, n = inputs[1].get_size()
        else:
            b, k, n = inputs[1].shape
        _, block_n, _ = micro_gemm.register_blocking
        padded_n = get_padded_n(n, block_n)

        W = inputs[1]
        new_inputs = list(inputs)
        if isinstance(W, ir.IRNode):
            new_size = [padded_n // block_n, k, block_n]
            if not isinstance(W, ir.TensorBox):
                W = ir.TensorBox(W)
            blocked_w = L.constant_pad_nd(W, (0, padded_n - n))
            blocked_w = L.permute(
                L.view(blocked_w, (b, k, padded_n // block_n, block_n)),
                [0, 2, 1, 3],
            )
            assert micro_gemm.get_b_layout() != LayoutType.NORMAL
            vnni_size = 4 if micro_gemm.get_b_layout() == LayoutType.VNNI4 else 2
            blocked_w = L.view(
                L.permute(
                    L.view(
                        blocked_w,
                        (b, padded_n // block_n, k // vnni_size, vnni_size, block_n),
                    ),
                    [0, 1, 2, 4, 3],
                ),
                [b] + new_size,
            )
            blocked_w = ir.ExternKernel.realize_input(blocked_w)
            blocked_w = ir.ExternKernel.require_contiguous(blocked_w)
            if isinstance(blocked_w, ir.ReinterpretView):
                # normalize stride to be "contiguous_strides" per size
                # this avoids the problems in L.view during template codegen
                assert isinstance(blocked_w.layout, ir.FixedLayout)
                blocked_w.layout = ir.FixedLayout(
                    blocked_w.layout.device,
                    blocked_w.layout.dtype,
                    blocked_w.layout.size,
                    ir.FlexibleLayout.contiguous_strides(blocked_w.layout.size),
                    blocked_w.layout.offset,
                )
        else:
            blocked_w = (
                torch.nn.functional.pad(W, (0, padded_n - n))
                .reshape(-1, k, padded_n // block_n, block_n)
                .transpose(1, 2)
                .contiguous()
            )
            assert micro_gemm.get_b_layout() != LayoutType.NORMAL
            layout_str = (
                "VNNI4" if micro_gemm.get_b_layout() == LayoutType.VNNI4 else "VNNI2"
            )
            assert micro_gemm.get_b_layout() in [
                LayoutType.VNNI2,
                LayoutType.VNNI4,
            ], f"We only support {layout_str} for now"
            vnni_size = 4 if micro_gemm.get_b_layout() == LayoutType.VNNI4 else 2
            assert (
                k % vnni_size == 0
            ), f"k should be divisible by vnni_size for {layout_str} layout"
            blocked_w = (
                blocked_w.view(
                    -1, padded_n // block_n, k // vnni_size, vnni_size, block_n
                )
                .transpose(-1, -2)
                .contiguous()
                .view(-1, padded_n // block_n, k, block_n)
            )
            # normalize stride to be "contiguous_strides" per size
            # this avoids the problems in L.view during template codegen
            new_stride = [1]
            for sz in reversed(blocked_w.shape[1:]):
                new_stride.insert(0, new_stride[0] * sz)
            blocked_w = blocked_w.as_strided(blocked_w.shape, new_stride)
        new_inputs[1] = blocked_w
        return new_inputs, layout_or_out

    def _get_default_reindexers(self, epilogue_nodes):
        def reindexer(args):
            if len(epilogue_nodes) == 0:
                return args
            return [0] + args

        return [reindexer]

    def get_options(self, kernel, template_buffer_node, epilogue_nodes, **kwargs):
        options, fake_buffers = super().get_options(
            kernel, template_buffer_node, epilogue_nodes, **kwargs
        )
        if options["micro_gemm"].get_b_layout() != LayoutType.NORMAL:
            self.should_pack_weights = True

        BX, BW, BY = options["X"], options["W"], options["Y"]
        options["BX"], options["BW"], options["BY"] = BX, BW, BY
        for kword in ["X", "W", "Y", "GemmOut", "Y_2d"]:
            options[kword] = kernel.select(options[kword], 0, 0)
        for kword in ["X", "W", "Y"]:
            options[kword + "_dtype"] = DTYPE_TO_CPP[options[kword].dtype]
        return options, fake_buffers

    def render(  # type: ignore[override]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        options, fake_buffers = self.get_options(
            kernel, template_buffer_node, epilogue_nodes, **kwargs
        )
        BX, BW, BY = options["BX"], options["BW"], options["BY"]
        X, W, Y = options["X"], options["W"], options["Y"]
        aliases = options["aliases"]

        with contextlib.ExitStack() as stack:
            for buf in fake_buffers:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )
            kernel.set_args(inputs={"X": X, "W": W}, outputs={"Y": Y}, aliases=aliases)
            result = self._template_from_string(MICROKERNEL_DEF).render(**options)
            result += self._template_from_string(
                GEMM_THREADED_MM_STUB + GEMM_TEMPLATE
            ).render(**options)
            result += self._template_from_string(
                GEMM_SINGLE_THREAD_MM_STUB + GEMM_TEMPLATE
            ).render(**{**options, "num_threads": 1})
            kernel.set_args(
                inputs={"BX": BX, "BW": BW}, outputs={"BY": BY}, aliases=aliases
            )
            result += self._template_from_string(BMM_WRAPPER).render(**options)
            return result
