import functools
import time
from functools import cached_property
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch._inductor.utils import is_cpu_device
from typing_extensions import Self


class LazyBenchmark:
    def __init__(self, initialize: Callable[[], float]) -> None:
        self.initialize = initialize

    @cached_property
    def value(self) -> float:
        return self.initialize()

    def __float__(self) -> float:
        return self.value
    
    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, format_spec: str) -> str:
        return self.value.__format__(format_spec)

    def __lt__(self, other: Any) -> bool:
        return other > self.value

    def __le__(self, other: Any) -> bool:
        return other >= self.value

    def __gt__(self, other: Any) -> bool:
        return other < self.value

    def __ge__(self, other: Any) -> bool:
        return other <= self.value

    def __add__(self, other: Any) -> Self:
        return LazyBenchmark(lambda: self.value + other)

    def __radd__(self, other: Any) -> Self:
        return LazyBenchmark(lambda: other + self.value)

    def __sub__(self, other: Any) -> Self:
        return LazyBenchmark(lambda: self.value - other)

    def __rsub__(self, other: Any) -> Self:
        return LazyBenchmark(lambda: other - self.value)

    def __mul__(self, other: Any) -> Self:
        return LazyBenchmark(lambda: self.value * other)

    def __rmul__(self, other: Any) -> Self:
        return LazyBenchmark(lambda: other * self.value)

    def __truediv__(self, other: Any) -> Self:
        return LazyBenchmark(lambda: self.value / other)

    def __rtruediv__(self, other: Any) -> Self:
        return LazyBenchmark(lambda: other / self.value)


class Benchmarker:
    def __init__(self) -> None:
        self.memory_cache: Dict[str, float] = {}
        self.kwargs_hash_to_futures_gpu: Dict[str, Tuple[LazyBenchmark, Callable[..., Any]]] = {}

    def benchmark(self, fn: Callable[..., Any], fn_args: List[Any], fn_kwargs: Dict[str, Any], **kwargs: Dict[str, Any]) -> float:
        _callable = lambda: fn(*fn_args, **fn_kwargs)

        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.benchmark_cpu(_callable, **kwargs)
        else:
            return self.benchmark_gpu(_callable, **kwargs)
    
    def benchmark_cpu(self, _callable: Callable[[], Any], warmup_iters: int = 5, benchmark_iters: int = 20) -> float:
        # this function borrowed from original implementation in torch._inductor.runtime.runtime_utils

        timings = []

        for _ in range(warmup_iters):
            _callable()
        for _ in range(benchmark_iters):
            start_time = time.perf_counter()
            _callable()
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)
        
        sorted_timings = sorted(timings)
        if benchmark_iters % 2  == 0:
            lower_timing = sorted_timings[(benchmark_iters // 2) - 1]
            upper_timing = sorted_timings[benchmark_iters // 2]
            timing = (lower_timing + upper_timing) / 2
        else:
            timing = sorted_timings[benchmark_iters // 2]

        return timing
    
    def benchmark_gpu(self, _callable: Callable[[], Any], estimation_iters: int = 5, memory_warmup_iters: int = 100, benchmark_iters: int = 100, max_benchmark_duration: int = 25) -> float:        
        def benchmark(buffer, _callable, iters):
            event_pairs = self.get_event_pairs(iters)

            start_time = time.perf_counter()
            for start_event, end_event in event_pairs:
                buffer.zero_()
                start_event.record()
                _callable()
                end_event.record()
            end_time = time.perf_counter()

            torch.cuda.synchronize()

            timing = self.get_min_timing(event_pairs)
            launch_overhead = (end_time - start_time) / iters

            return timing, launch_overhead
        
        def calculate_required_gpu_sleep_cycles(launch_overhead, memory_warmup_iters, benchmark_iters) -> int:
            memory_warmup_overhead = self.get_launch_overhead_per_cache_clear() * memory_warmup_iters
            benchmarking_overhead = launch_overhead * benchmark_iters
            required_sleep_cycles = (memory_warmup_overhead + benchmarking_overhead) / self.get_time_per_gpu_sleep_cycle()
            return int(required_sleep_cycles)
        
        # make sure these functions are initialized, do it before we initialize buffer and
        # after we're sure that we're on GPU. launch in this specific order to minimize
        # potential overhead, as self.get_launch_overhead_per_cache_clear() doesn't
        # call torch.cuda.synchronize() on exit
        self.get_launch_overhead_per_cache_clear()
        self.get_cache_size()
        self.get_time_per_gpu_sleep_cycle()
        
        # initialize _callable, usually the first call is significantly slower than others
        try:
            _callable()
        except Exception:
            return float("inf")

        # initialize buffer, like _callable usually the first buffer.zero_() is the slowest
        buffer = torch.empty(int(self.get_cache_size() // 4), dtype=torch.int, device="cuda")
        buffer.zero_()

        # estimate the running time of _callable and shrink the size of benchmark_iters to ensure that
        # the benchmarking finishes in roughly max_benchmark_duration or less. we also take advantage of
        # the estimation loop to measure the launch overhead of _callable
        estimated_timing, launch_overhead = benchmark(buffer, _callable, estimation_iters)
        benchmark_iters = min(benchmark_iters, max(int(max_benchmark_duration / estimated_timing), 1))

        # calculate the number of GPU sleep cycles required to completely overlap the overhead of
        # memory_warmup_iters and benchmark_iters. this can reduce measurement inaccuracy by preloading
        # the stream with a large number of events
        required_gpu_sleep_cycles = calculate_required_gpu_sleep_cycles(launch_overhead, memory_warmup_iters, benchmark_iters)
        torch.cuda._sleep(required_gpu_sleep_cycles)

        for _ in range(memory_warmup_iters):
            buffer.zero_()
        timing, _ = benchmark(buffer, _callable, benchmark_iters)
        
        del buffer

        return timing
    
    def benchmark_many_gpu(self, callables: Callable[[], Any], estimation_iters: int = 5, memory_warmup_iters: int = 100, benchmark_iters: int = 100, max_benchmark_duration: int = 25) -> List[float]:
        callable_to_timing = {}

        for _callable in callables:
            try:
                _callable()
            except Exception:
                callable_to_timing[_callable] = float("inf")
        
        callables_to_benchmark = [_callable for _callable in callables if _callable not in callable_to_timing]
        for _callable in callables_to_benchmark:
            callable_to_timing[_callable] = self.benchmark_gpu(_callable, estimation_iters=estimation_iters, memory_warmup_iters=memory_warmup_iters)
        
        return [callable_to_timing[_callable] for _callable in callables]

    def lazy_benchmark(self, fn: Callable[..., Any], fn_args: List[Any], fn_kwargs: Dict[str, Any], **kwargs: Dict[str, Any]) -> Union[LazyBenchmark, float]:
        _callable = lambda: fn(*fn_args, **fn_kwargs)

        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.lazy_benchmark_cpu(_callable, **kwargs)
        else:
            return self.lazy_benchmark_gpu(_callable, **kwargs)
    
    def lazy_benchmark_cpu(self, _callable: Callable[[], Any], **kwargs: Dict[str, Any]) -> float:
        return self.benchmark_cpu(_callable, **kwargs)

    def lazy_benchmark_gpu(self, _callable: Callable[[], Any], **kwargs: Dict[str, Any]) -> LazyBenchmark:
        kwargs_hash = hash(tuple(sorted(kwargs.items())))
        this_key = hash(_callable) + kwargs_hash

        self.kwargs_hash_to_futures_gpu[kwargs_hash] = self.kwargs_hash_to_futures_gpu.get(kwargs_hash, []) + [(_callable, this_key)]

        def initialize():
            if this_key in self.memory_cache:
                return self.memory_cache[this_key]

            futures_gpu = self.kwargs_hash_to_futures_gpu.pop(kwargs_hash)

            callables, keys = zip(*futures_gpu)
            callables, keys = list(callables), list(keys)

            timings = self.benchmark_many_gpu(callables, **kwargs)

            for key, timing in zip(keys, timings):
                self.memory_cache[key] = timing
            
            return self.memory_cache[key]
        
        return LazyBenchmark(initialize)

    @functools.lru_cache(None)
    def get_cache_size(self) -> int:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return properties.l2CacheSize
    
    @functools.lru_cache(None)
    def get_time_per_gpu_sleep_cycle(self) -> float:
        torch.cuda.synchronize()

        gpu_sleep_cycles = 1000000

        start_time = time.perf_counter()
        torch.cuda._sleep(gpu_sleep_cycles)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        return (end_time - start_time) / gpu_sleep_cycles
    
    @functools.lru_cache(None)
    def get_launch_overhead_per_cache_clear(self) -> float:
        torch.cuda.synchronize()

        buffer = torch.empty(int(self.get_cache_size() // 4), dtype=torch.int, device="cuda")
        cache_clear_iters = 100

        start_time = time.perf_counter()
        for _ in range(cache_clear_iters):
            buffer.zero_()
        end_time = time.perf_counter()

        del buffer

        return (end_time - start_time) / cache_clear_iters

    def get_event_pairs(self, num_pairs: int) -> List[Tuple[torch.cuda.Event, torch.cuda.Event]]:
        return [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(num_pairs)
        ]
    
    def get_interleaved_event_pairs(self, num_callables: int, num_pairs: int) -> List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]]:
        return [self.get_event_pairs(num_callables) for _ in range(num_pairs)]

    def get_min_timing(self, event_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
        return min([start_event.elapsed_time(end_event) for start_event, end_event in event_pairs])
    
    def get_interleaved_min_timing(self, interleaved_event_pairs: List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]]) -> float:
        return [self.get_min_timing(event_pairs) for event_pairs in zip(*interleaved_event_pairs)]

    
benchmarker = Benchmarker()
