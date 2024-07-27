# flake8: noqa: B950
# fmt: off
from typing import List, Optional, Tuple

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import (
    LearnedHeuristicDecision,
)

class MixedMMH100(LearnedHeuristicDecision):

    def __init__(self) -> None:
        self.choices: List[Choice] = []
        self.fill_choices()

    def check_precondition(self, metadata: AHMetadata, context: AHContext,) -> bool:
        return (
            metadata.name == self.get_name()
            and metadata.shared_memory == 232448
            and str(metadata.device_capa) == "(9, 0)"
        )

    def get_confidence_threshold(self) -> float:
        return 0.0

    def get_choice(self, idx: int) -> Optional[str]:
        if idx < len(self.choices):
            return self.choices[idx]
        return None

    def fill_choices(self) -> None:
        self.choices.append('extern_fallback_mixed_mm')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=256_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=256_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=8')

    def get_name(self) -> str:
        return 'mixed_mm'

    def get_best_choices(self, context: AHContext) -> Optional[List[Tuple[float, int]]]:
        if context.get_value('arith_intensity') <= 15.988086223602295:
            if context.get_value('n') <= 25280.0:
                if context.get_value('n') <= 1344.0:
                    if context.get_value('mat1_stride_0') <= 7808.0:
                        return [(0.581, 7), (0.419, 6)]
                    else:
                        if context.get_value('m*n') <= 7680.0:
                            return [(0.875, 0), (0.125, 6)]
                        else:
                            return [(0.833, 0), (0.167, 7)]
                else:
                    if context.get_value('n') <= 8512.0:
                        if str(context.get_value('mat2_dtype')) != 'torch.int8':
                            return [(0.763, 6), (0.237, 7)]
                        else:
                            return [(0.725, 7), (0.275, 6)]
                    else:
                        if str(context.get_value('mat1_dtype')) != 'torch.bfloat16':
                            return [(0.736, 7), (0.197, 9), (0.048, 6), (0.014, 8), (0.005, 10)]
                        else:
                            return [(0.473, 7), (0.398, 6), (0.097, 9), (0.032, 10)]
            else:
                if context.get_value('n') <= 42254.0:
                    if context.get_value('n') <= 33856.0:
                        if context.get_value('k*n') <= 68157440.0:
                            return [(0.370, 4), (0.370, 5), (0.074, 7), (0.074, 8), (0.074, 11), (0.037, 6)]
                        else:
                            return [(0.916, 8), (0.036, 7), (0.036, 9), (0.012, 4)]
                    else:
                        return [(0.659, 5), (0.341, 6)]
                else:
                    if context.get_value('k*n') <= 326052992.0:
                        if context.get_value('n') <= 55232.0:
                            return [(0.571, 6), (0.321, 7), (0.036, 4), (0.036, 8), (0.036, 9)]
                        else:
                            return [(0.506, 6), (0.325, 8), (0.104, 7), (0.039, 5), (0.026, 9)]
                    else:
                        if context.get_value('n') <= 57024.0:
                            return [(0.462, 9), (0.385, 7), (0.115, 6), (0.038, 8)]
                        else:
                            return [(0.598, 8), (0.223, 9), (0.107, 6), (0.071, 7)]
        else:
            if context.get_value('m*n') <= 543936.0:
                if str(context.get_value('17LEQmLEQ32')) != 'True':
                    if context.get_value('m*n') <= 262272.0:
                        if context.get_value('n') <= 1592.5:
                            return [(0.860, 0), (0.140, 9)]
                        else:
                            return None
                    else:
                        if context.get_value('m*k') <= 1294336.0:
                            return [(0.833, 17), (0.150, 18), (0.017, 15)]
                        else:
                            return [(0.917, 17), (0.083, 8)]
                else:
                    if context.get_value('n') <= 12416.0:
                        if context.get_value('m*n') <= 43008.0:
                            return None
                        else:
                            return [(0.853, 14), (0.147, 9)]
                    else:
                        return [(0.625, 12), (0.375, 14)]
            else:
                if context.get_value('m') <= 32.5:
                    if context.get_value('mat2_stride_1') <= 6656.0:
                        if context.get_value('n') <= 69184.0:
                            return [(0.611, 12), (0.361, 14), (0.028, 13)]
                        else:
                            return [(1.000, 12)]
                    else:
                        if context.get_value('mat2_stride_1') <= 20864.0:
                            return [(1.000, 12)]
                        else:
                            return [(0.958, 12), (0.042, 9)]
                else:
                    if context.get_value('m*n') <= 1085440.0:
                        if context.get_value('n') <= 9152.0:
                            return [(1.000, 18)]
                        else:
                            return [(0.780, 18), (0.160, 16), (0.060, 20)]
                    else:
                        if context.get_value('m') <= 67.0:
                            return [(0.650, 16), (0.203, 19), (0.122, 18), (0.016, 20), (0.008, 1)]
                        else:
                            return [(0.561, 3), (0.185, 16), (0.096, 20), (0.083, 19), (0.076, 2)]