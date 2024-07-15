from collections import defaultdict
from typing import Optional
import functools
import time
from functools import cached_property
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch._inductor.utils import is_cpu_device


class LazyBenchmark:
    def __init__(self, initialize: Callable[[], float]) -> None:
        self.initialize = initialize

    @cached_property
    def value(self) -> float:
        return self.initialize()
    
    __float__ = lambda self: self.value
    __format__ = lambda self, format_spec: format(self.value, format_spec)
    __str__ = lambda self: str(self.value)
    
    __lt__ = lambda self, other: other > self.value
    __le__ = lambda self, other: other >= self.value

    __gt__ = lambda self, other: other < self.value
    __ge__ = lambda self, other: other <= self.value
    
    __add__ = lambda self, other: LazyBenchmark(lambda: self.value + other)
    __radd__ = lambda self, other: LazyBenchmark(lambda: other + self.value)

    __sub__ = lambda self, other: LazyBenchmark(lambda: self.value - other)
    __rsub__ = lambda self, other: LazyBenchmark(lambda: other - self.value)

    __mul__ = lambda self, other: LazyBenchmark(lambda: self.value * other)
    __rmul__ = lambda self, other: LazyBenchmark(lambda: other * self.value)

    __truediv__ = lambda self, other: LazyBenchmark(lambda: self.value / other)
    __rtruediv__ = lambda self, other: LazyBenchmark(lambda: other / self.value)


class Benchmarker:
    def __init__(self) -> None:
        self.memory_cache: Dict[str, Optional[float]] = defaultdict(lambda: None)

        self.kwargs_hash_to_futures_gpu: Dict[str, Tuple[LazyBenchmark, Callable[..., Any]]] = {}

    def benchmark(self, fn: Callable[..., Any], fn_args: List[Any], fn_kwargs: Dict[str, Any], **kwargs: Dict[str, Any]) -> float:
        _callable = lambda: fn(*fn_args, **fn_kwargs)

        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.benchmark_cpu(_callable, **kwargs)
        else:
            return self.benchmark_gpu(_callable, **kwargs)
    
    def benchmark_cpu(self, _callable: Callable[[], Any], warmup_iters: int = 5, benchmark_iters: int = 20) -> float:
        def benchmark(_callable, iters):
            timings = []

            for _ in range(iters):
                start_time = time.perf_counter()
                _callable()
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)
            
            return timings
        
        def get_median_timing(timings):
            timings = sorted(timings)

            if ((len(timings) % 2) == 0):
                lower_timing = timings[(len(timings) // 2) - 1]
                upper_timing = timings[len(timings) // 2]
                median_timing = (lower_timing + upper_timing) / 2
            else:
                median_timing = timings[len(timings) // 2]
            
            return median_timing

        for _ in range(warmup_iters):
            _callable()
        
        timings = benchmark(_callable, benchmark_iters)

        timing = get_median_timing(timings)

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
            cpu_launch_overhead_per_iter = (end_time - start_time) / iters

            return timing, cpu_launch_overhead_per_iter
                
        try:
            _callable()
        except Exception:
            return float("inf")

        buffer = torch.empty(int(self.get_cache_size() // 4), dtype=torch.int, device="cuda")
        buffer.zero_()

        estimated_timing, cpu_launch_overhead_per_iter = benchmark(buffer, _callable, estimation_iters)

        benchmark_iters = min(benchmark_iters, max(int(max_benchmark_duration / estimated_timing), 1))

        torch.cuda._sleep(
            int(
                (
                    (self.get_cpu_launch_overhead_per_gpu_cache_clear() * memory_warmup_iters)
                    + (cpu_launch_overhead_per_iter * benchmark_iters)
                )
                / self.get_gpu_time_per_clock_cycle()
            )
        )

        for _ in range(memory_warmup_iters):
            buffer.zero_()

        timing, _ = benchmark(buffer, _callable, benchmark_iters)
        
        del buffer

        return timing
    
    def benchmark_many_gpu(self, callables: List[Callable[[], Any]], estimation_iters: int = 5, memory_warmup_iters: int = 100, benchmark_iters: int = 100, max_benchmark_duration: int = 25) -> List[float]:        
        def benchmark(buffer, callables, iters):
            interleaved_event_pairs = self.get_interleaved_event_pairs(len(callables), iters)

            start_time = time.perf_counter()
            for event_pairs in interleaved_event_pairs:
                for _callable, (start_event, end_event) in zip(callables, event_pairs):
                    buffer.zero_()
                    start_event.record()
                    _callable()
                    end_event.record()
            end_time = time.perf_counter()

            torch.cuda.synchronize()

            timings = self.get_interleaved_min_timings(interleaved_event_pairs)
            cpu_launch_overhead_per_iter = (end_time - start_time) / iters

            return timings, cpu_launch_overhead_per_iter

        callable_to_timing = {}
        callables_to_benchmark = []

        for _callable in callables:
            try:
                _callable()
            except Exception:
                callable_to_timing[_callable] = float("inf")
            else:
                callables_to_benchmark.append(_callable)
    
        buffer = torch.empty(int(self.get_cache_size() // 4), dtype=torch.int, device="cuda")
        buffer.zero_()

        estimated_timings, cpu_launch_overhead_per_iter = benchmark(buffer, callables_to_benchmark, estimation_iters)

        benchmark_iters = min(benchmark_iters, max(int(max_benchmark_duration / max(estimated_timings)), 1))

        torch.cuda._sleep(
            int(
                (
                    (self.get_cpu_launch_overhead_per_gpu_cache_clear() * memory_warmup_iters)
                    + (cpu_launch_overhead_per_iter * benchmark_iters)
                )
                / self.get_gpu_time_per_clock_cycle()
            )
        )

        for _ in range(memory_warmup_iters):
            buffer.zero_()

        timings, _ = benchmark(buffer, callables_to_benchmark, benchmark_iters)
        
        del buffer

        for _callable, timing in zip(callables_to_benchmark, timings):
            callable_to_timing[_callable] = timing
        
        timings = [callable_to_timing[_callable] for _callable in callables]

        return timings

    def lazy_benchmark(self, fn: Callable[..., Any], fn_args: List[Any], fn_kwargs: Dict[str, Any], **kwargs: Dict[str, Any]) -> LazyBenchmark:
        _callable = lambda: fn(*fn_args, **fn_kwargs)

        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.lazy_benchmark_cpu(_callable, **kwargs)
        else:
            return self.lazy_benchmark_gpu(_callable, **kwargs)
    
    def lazy_benchmark_cpu(self, _callable: Callable[[], Any], **kwargs: Dict[str, Any]) -> LazyBenchmark:
        lazy_benchmark = LazyBenchmark(lambda: self.benchmark_cpu(_callable, **kwargs))
        return lazy_benchmark

    def lazy_benchmark_gpu(self, _callable: Callable[[], Any], **kwargs: Dict[str, Any]) -> LazyBenchmark:
        kwargs_hash = hash(tuple(sorted(kwargs.items())))

        key = hash(_callable) + kwargs_hash

        self.kwargs_hash_to_futures_gpu[kwargs_hash] = self.kwargs_hash_to_futures_gpu.get(kwargs_hash, []) + [(_callable, key)]

        def initialize() -> float:
            if key in self.memory_cache:
                return self.memory_cache[key]

            futures_gpu = self.kwargs_hash_to_futures_gpu.pop(kwargs_hash)

            callables, keys = zip(*futures_gpu)
            callables, keys = list(callables), list(keys)

            timings = self.benchmark_many_gpu(callables, **kwargs)

            self.memory_cache.update(zip(keys, timings))
            
            return self.memory_cache[key]
        
        return LazyBenchmark(initialize)

    @functools.lru_cache(None)
    def get_cache_size(self) -> int:
        return 50 * 1024 * 1024
    
    @functools.lru_cache(None)
    def get_gpu_time_per_clock_cycle(self, cycles_to_sleep: int = 10000000) -> float:
        torch.cuda.synchronize()

        start_time = time.perf_counter()
        torch.cuda._sleep(cycles_to_sleep)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        return (end_time - start_time) / cycles_to_sleep
    
    @functools.lru_cache(None)
    def get_cpu_launch_overhead_per_gpu_cache_clear(self, cache_clear_iters: int = 1000) -> float:
        buffer = torch.empty(int(self.get_cache_size() // 4), dtype=torch.int, device="cuda")

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
    
    def get_interleaved_min_timings(self, interleaved_event_pairs: List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]]) -> float:
        return [self.get_min_timing(event_pairs) for event_pairs in zip(*interleaved_event_pairs)]

    
benchmarker = Benchmarker()
