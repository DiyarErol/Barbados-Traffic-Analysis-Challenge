"""
Performance Benchmark Suite
===========================
Monitor CPU, memory, GPU usage and performance metrics.
"""

import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    
    name: str
    duration: float  # seconds
    cpu_percent: float
    memory_mb: float
    memory_peak_mb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    throughput: Optional[float] = None  # samples/second
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'duration_seconds': self.duration,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'gpu_utilization': self.gpu_utilization,
            'gpu_memory_mb': self.gpu_memory_mb,
            'throughput_samples_per_sec': self.throughput,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"\n{'='*60}",
            f"Benchmark: {self.name}",
            f"{'='*60}",
            f"Duration:      {self.duration:.2f}s",
            f"CPU Usage:     {self.cpu_percent:.1f}%",
            f"Memory:        {self.memory_mb:.1f} MB (Peak: {self.memory_peak_mb:.1f} MB)",
        ]
        
        if self.gpu_utilization is not None:
            lines.append(f"GPU Usage:     {self.gpu_utilization:.1f}%")
        
        if self.gpu_memory_mb is not None:
            lines.append(f"GPU Memory:    {self.gpu_memory_mb:.1f} MB")
        
        if self.throughput is not None:
            lines.append(f"Throughput:    {self.throughput:.1f} samples/sec")
        
        lines.append(f"{'='*60}\n")
        
        return '\n'.join(lines)


class PerformanceBenchmark:
    """Monitor and benchmark performance."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize performance benchmark.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir or Path("benchmarks/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: list[BenchmarkResult] = []
        self.process = psutil.Process()
    
    def benchmark(self, func: Callable, *args, 
                  name: str = "Benchmark",
                  n_samples: Optional[int] = None,
                  **kwargs) -> BenchmarkResult:
        """
        Benchmark a function.
        
        Args:
            func: Function to benchmark
            *args: Function arguments
            name: Benchmark name
            n_samples: Number of samples (for throughput calculation)
            **kwargs: Function keyword arguments
            
        Returns:
            BenchmarkResult
        """
        print(f"\nRunning benchmark: {name}...")
        
        # Reset CPU percent measurement
        self.process.cpu_percent()
        
        # Get initial memory
        mem_before = self.process.memory_info().rss / 1024 / 1024
        peak_memory = mem_before
        
        # Get initial GPU stats
        gpu_util_before = None
        gpu_mem_before = None
        if HAS_GPU:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util_before = gpus[0].load * 100
                    gpu_mem_before = gpus[0].memoryUsed
            except:
                pass
        
        # Run benchmark
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        # Get final stats
        cpu_percent = self.process.cpu_percent()
        mem_after = self.process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, mem_after)
        
        gpu_util_after = None
        gpu_mem_after = None
        if HAS_GPU:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util_after = gpus[0].load * 100
                    gpu_mem_after = gpus[0].memoryUsed
            except:
                pass
        
        # Calculate throughput
        throughput = None
        if n_samples is not None and duration > 0:
            throughput = n_samples / duration
        
        # Create result
        benchmark_result = BenchmarkResult(
            name=name,
            duration=duration,
            cpu_percent=cpu_percent,
            memory_mb=mem_after,
            memory_peak_mb=peak_memory,
            gpu_utilization=gpu_util_after,
            gpu_memory_mb=gpu_mem_after,
            throughput=throughput
        )
        
        self.results.append(benchmark_result)
        
        print(benchmark_result)
        
        return benchmark_result
    
    def benchmark_video_processing(self, video_extractor, video_paths: list):
        """Benchmark video feature extraction."""
        return self.benchmark(
            video_extractor.extract_batch,
            video_paths,
            show_progress=False,
            name="Video Feature Extraction",
            n_samples=len(video_paths)
        )
    
    def benchmark_model_training(self, trainer, X_train, y_train):
        """Benchmark model training."""
        return self.benchmark(
            trainer.train,
            X_train,
            y_train,
            name="Model Training",
            n_samples=len(X_train)
        )
    
    def benchmark_inference(self, predictor, X_test):
        """Benchmark model inference."""
        return self.benchmark(
            predictor.predict,
            X_test,
            name="Model Inference",
            n_samples=len(X_test)
        )
    
    def compare_results(self) -> pd.DataFrame:
        """
        Compare all benchmark results.
        
        Returns:
            DataFrame with comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        print(f"\n{'='*80}")
        print("Benchmark Comparison")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        return df
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_path = self.output_dir / filename
        
        results_data = {
            'benchmarks': [r.to_dict() for r in self.results],
            'system_info': self._get_system_info()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n✓ Benchmark results saved to: {output_path}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'python_version': psutil.__version__
        }
        
        if HAS_GPU:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    info['gpu_name'] = gpus[0].name
                    info['gpu_memory_gb'] = gpus[0].memoryTotal / 1024
            except:
                pass
        
        return info
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate markdown report."""
        if output_path is None:
            output_path = self.output_dir / "benchmark_report.md"
        
        lines = [
            "# Performance Benchmark Report\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## System Information\n"
        ]
        
        sys_info = self._get_system_info()
        for key, value in sys_info.items():
            lines.append(f"- **{key}:** {value}")
        
        lines.append("\n## Benchmark Results\n")
        
        if self.results:
            df = self.compare_results()
            lines.append("```")
            lines.append(df.to_string(index=False))
            lines.append("```\n")
            
            lines.append("## Detailed Results\n")
            for result in self.results:
                lines.append(f"### {result.name}\n")
                lines.append(f"- **Duration:** {result.duration:.2f}s")
                lines.append(f"- **CPU Usage:** {result.cpu_percent:.1f}%")
                lines.append(f"- **Memory:** {result.memory_mb:.1f} MB (Peak: {result.memory_peak_mb:.1f} MB)")
                if result.throughput:
                    lines.append(f"- **Throughput:** {result.throughput:.1f} samples/sec")
                lines.append("")
        
        report = '\n'.join(lines)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Benchmark report saved to: {output_path}")
        
        return report


def run_full_benchmark():
    """Run comprehensive benchmark suite."""
    print(f"\n{'#'*80}")
    print("# PERFORMANCE BENCHMARK SUITE")
    print(f"{'#'*80}\n")
    
    benchmark = PerformanceBenchmark()
    
    # System info
    print("\nSystem Information:")
    for key, value in benchmark._get_system_info().items():
        print(f"  {key}: {value}")
    
    # Example benchmarks (customize as needed)
    print("\n" + "="*80)
    print("Running benchmarks...")
    print("="*80)
    
    # Memory stress test
    def memory_test():
        # Simulate large array operations
        arr = np.random.rand(10000, 1000)
        return arr.mean()
    
    benchmark.benchmark(memory_test, name="Memory Stress Test")
    
    # CPU stress test
    def cpu_test():
        # Simulate computation
        result = 0
        for i in range(1000000):
            result += i ** 2
        return result
    
    benchmark.benchmark(cpu_test, name="CPU Stress Test")
    
    # Save and report
    benchmark.save_results()
    benchmark.generate_report()
    benchmark.compare_results()
    
    print(f"\n{'#'*80}")
    print("# BENCHMARK COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    run_full_benchmark()
